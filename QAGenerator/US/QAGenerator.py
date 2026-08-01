import inspect
import json
import string
import random
import pandas as pd

from typing import Optional
from tqdm import tqdm

from QATemplate import *
from QACalculator import *

import warnings
warnings.filterwarnings("ignore")



def check_df_fundamental(df):
    if df.isna().mean().mean() >= 0.4:  # If nan values are too many, trigger resampling
        raise IndicatorError('Too much missing data')
    for col in ['PretaxIncomeLoss', 'Revenue', 'CostOfGoodsAndServicesSold', 'PreferredDividendExpense', 'AllEquityBalanceIncludingMinorityInterest', 'NetIncome', 'Liabilities']:
        if col in df.columns and not df[col].fillna(0).gt(0).all():
            raise IndicatorError(f'{col} must be positive')

def get_one_data_df_fundamental(corporate, indicator, start_year, end_year):
    df = pd.read_csv(f'./data/{corporate}/fundamental.csv')

    # Indicators with shift(1) require data from the previous year, so subtract 1 from start_year
    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function'])
    ):
        start_year -= 1

    if start_year < 2016:
        raise IndicatorError(f'Start year {start_year} too early!')

    df = df.set_index('year').sort_index()
    df = df.query(f'year >= {start_year} and year <= {end_year}')
    if len(df) != (end_year - start_year + 1):
        raise IndicatorError(f'Not enough data for {corporate} from {start_year} to {end_year}')

    if 'AmortizationOfIntangibles' in indicator['data'] and 'AmortizationOfIntangibles' not in df.columns:
        df['AmortizationOfIntangibles'] = 0

    try:
        df = df.loc[:, indicator['data']]
    except KeyError as e:
        print(f'Missing required data for {corporate} from {start_year} to {end_year}: {e}')
        raise IndicatorError(f'Missing required data: {e}')

    # These fields are sometimes reported as negative (deduction sign) by some companies
    for col in ('CapitalExpenses', 'InterestExpense', 'DepreciationExpense', 'AmortizationOfIntangibles', 'CostOfGoodsAndServicesSold', 'TotalOperatingExpenses'):
        if col in df.columns:
            df[col] = df[col].abs()

    check_df_fundamental(df)
    df = df.fillna(0)
    return df

def get_data_df_fundamental(corporates, indicator, start_year, end_year):
    if len(corporates) == 1:
        return get_one_data_df_fundamental(corporates[0], indicator, start_year, end_year)
    else:
        for corporate in corporates:
            for keyword in ['security', 'bank', 'insurance', 'fund', 'trust', 'investment', 'finance']:
                if keyword in corporate.lower():
                    raise IndicatorError('Financial institutions have too much missing data for fundamental indicators')
        df_list = []
        for corporate in corporates:
            df_corp = get_one_data_df_fundamental(corporate, indicator, start_year, end_year).reset_index()
            if df_corp.empty:
                raise IndicatorError(f'No data for {corporate} from {start_year} to {end_year}')
            df_corp.insert(0, 'corporate', corporate)
            df_list.append(df_corp)
        df = pd.concat(df_list, ignore_index=True).set_index(['corporate', 'year'])
        return df

def calculate_indicator_fundamental(df: pd.DataFrame, indicator: dict):
    df = indicator['function'](df)
    df = df.dropna()
    return df

def get_data_df_technical(corporate, indicator, start_date, end_date, n=None):
    df = pd.read_csv(f'./data/{corporate}/technical.csv', parse_dates=['date'])

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()

    if start_date == end_date:  # When calculating for a single point, check if it's a non-trading day
        if df.query(f'date == "{start_date}"').empty:
            return None  # Non-trading day

    if n is not None:  # Push start_date back by n trading days
        i_start_date = df.query(f'date >= "{start_date}"').index[0] - n
        if i_start_date < 0:
            raise IndicatorError('Not enough historical data for technical indicator')
        start_date = df['date'].iloc[i_start_date].date()

    df = df.query(f'date >= "{start_date}" and date <= "{end_date}"')

    df = df.set_index('date').sort_index()
    df = df.reindex(columns=indicator["data"])
    return df

def calculate_indicator_technical(df: pd.DataFrame, indicator: dict, n_list: Optional[list] = None):
    if n_list is None:
        result = indicator['function'](df)
        if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            return result.dropna()
        else:
            return result

    df_list = []
    for n in n_list:
        df_n = indicator['function'](df, n)
        df_n = df_n.to_frame(name=f"{n}-day")
        df_list.append(df_n)
    return pd.concat(df_list, axis=1).dropna()

def get_data_df_technical_performance(corporate, indicator, start_date, end_date):
    df = pd.read_csv(f'./data/{corporate}/technical.csv', parse_dates=['date'])

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()

    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function'])
    ):
        start_date = df['date'].iloc[df.query(f'date >= "{start_date}"').index[0] - 1].date()

    df = df.query(f'date >= "{start_date}" and date <= "{end_date}"')

    df = df.set_index('date').sort_index()
    df = df.reindex(columns=indicator["data"])

    if 'benchmark_close' in indicator['data'] or 'market_close' in indicator['data']:
        df_index = pd.read_csv(f'data/SP500.csv', parse_dates=['date']).set_index('date').sort_index()[['close']]
        df_index = df_index.query(f'date >= "{start_date}" and date <= "{end_date}"')
        if 'benchmark_close' in indicator['data']:
            df['benchmark_close'] = df_index
        if 'market_close' in indicator['data']:
            df['market_close'] = df_index
    return df

def get_data_df_macro(indicator, start_year, end_year, indicator_name=None):
    df = pd.read_csv(f'./data/macro.csv')

    # Indicators with shift(1) or pct_change require data from the previous year, so subtract 1 from start_year
    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function'])
    ):
        start_year -= 1
    
    if start_year < 2016:
        raise IndicatorError(f'Start year {start_year} too early!')

    df = df.query(f'year >= {start_year} and year <= {end_year}')
    df = df.set_index('year').sort_index()
    df = df.reindex(columns=indicator["data"])

    if df.isna().sum().sum() > 0:
        raise IndicatorError('Missing data for macro indicator')

    return df

def calculate_indicator_macro(df: pd.DataFrame, indicator: dict, indicator_name=None):
    df = indicator['function'](df)
    # Indicators with shift(1) or pct_change need the first row removed
    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function'])
    ):
        df = df.iloc[1:]
    return df


def build_corporate_pools(indicators):
    """Pre-compute valid corporates per indicator based on required column availability."""
    company_cols = {}
    for corporate in CORPORATE_LIST:
        cols = set(pd.read_csv(f'./data/{corporate}/fundamental.csv', nrows=0).columns)
        company_cols[corporate] = cols

    pools = {}
    for name, indicator in indicators.items():
        # AmortizationOfIntangibles has a zero-fill fallback — exclude from required check
        required = [c for c in indicator['data'] if c != 'AmortizationOfIntangibles']
        pools[name] = [c for c in CORPORATE_LIST if all(col in company_cols[c] for col in required)]
    return pools


def generate_fundamental(output_path=None, add_note=True, n_per=5):
    corporate_pools = build_corporate_pools(indicators_fundamental)

    var_pools = {
        'corporate': CORPORATE_LIST,
        'corporate1': CORPORATE_LIST,
        'corporate2': CORPORATE_LIST,
        'year': END_YEAR_LIST,
        'end_year': END_YEAR_LIST,
        't_year': T_YEAR_LIST,
        'quarter': QUARTER_LIST,
        'indicator': list(indicators_fundamental.keys()),
        'indicator1': list(indicators_fundamental.keys()),
        'indicator2': list(indicators_fundamental.keys()),
        'statistic': STATISTIC_LIST,
        'high_or_low': HIGH_OR_LOW_LIST,
        'change': CHANGE_LIST,
    }

    QAs = []
    for template_group in question_template_fundamental:
        templates = template_group['templates']
        function = template_group['function']
        difficulty = template_group['difficulty']
        print(f"Generating QAs for template group: {templates[0]}")
        required_vars = [v[1] for v in string.Formatter().parse(templates[0]) if v[1] is not None]
        iter_vars = list(required_vars)

        if 'start_year' in required_vars and 'end_year' in required_vars:
            iter_vars[required_vars.index('start_year')] = 't_year'

        if 'corporates' in iter_vars:
            iter_vars.remove('corporates')

        # Determine the main iteration variable (priority: indicator, then indicator1)
        main_iter_var = 'indicator' if 'indicator' in iter_vars else 'indicator1'
        if main_iter_var in iter_vars:
            iter_vars.remove(main_iter_var)

        # Iterate over all indicators
        for indicator_name in tqdm(var_pools[main_iter_var]):
            valid_corporates = corporate_pools.get(indicator_name, CORPORATE_LIST)
            for key in ('corporate', 'corporate1', 'corporate2'):
                var_pools[key] = valid_corporates

            used_corporates = set()
            while len(used_corporates) < n_per:
                question_template = templates[len(used_corporates) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                # Use a while loop to ensure a valid parameter combination (e.g., corporate1 != corporate2)
                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    if 'corporates' in required_vars:
                        kwargs['corporates'] = ', '.join(random.sample(valid_corporates, random.randint(3, 5)))
                        # kwargs['corporates'] = ', '.join(random.sample(CORPORATE_LIST, random.randint(2, 2)))

                    # Validation: remove invalid case where corporate1 == corporate2
                    if 'corporate1' in kwargs and 'corporate2' in kwargs:
                        if kwargs['corporate1'] == kwargs['corporate2']:
                            continue  # Continue while loop to resample

                    # Validation: when statistic == 'CAGR', indicator can only be specific ones
                    if 'statistic' in kwargs and kwargs['statistic'] == 'CAGR':
                        if kwargs[main_iter_var] not in ['Liability-to-Asset Ratio', 'Equity Multiplier', 'EBITDA', 'NOPAT', 'Operating Cash Flow Per Share', 'Free Cash Flow', 'Free Cash Flow to Firm', 'Free Cash Flow to Equity', 'Total Asset Turnover']:
                            continue  # Continue while loop to resample

                    # Calculate derived variables
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        kwargs['start_year'] = kwargs['end_year'] - kwargs['t_year']
                        kwargs.pop('t_year')

                    indicator = indicators_fundamental[kwargs[main_iter_var]]

                    if add_note:
                        note = f' ({indicator["note"]})' if 'note' in indicator else ''
                    else:
                        note = ''

                    if 'what is the ratio' in templates[0] or 'How many times' in templates[0]:
                        note = note.replace('Unit:', 'Indicator unit:')

                    question = question_template.format(**kwargs) + note + question_suffix_fundamental
                    # print(question)  ### debug

                    # Get raw data
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        start_year = kwargs['start_year']
                        end_year = kwargs['end_year']
                    if 'year' in required_vars:
                        start_year = end_year = kwargs['year']

                    if 'corporates' in required_vars:
                        corporates = kwargs['corporates'].split(', ')
                    elif 'corporate' in required_vars:
                        corporates = [kwargs['corporate']]
                    elif 'corporate1' in required_vars and 'corporate2' in required_vars:
                        corporates = [kwargs['corporate1'], kwargs['corporate2']]
                    else:
                        raise IndicatorError('No corporate variable found in question template.')

                    # Validation: sampled corporates must not overlap with already-used ones
                    if {tuple(corporates)} & used_corporates:
                        continue

                    try:
                        data = get_data_df_fundamental(corporates, indicator, start_year, end_year)
                    except IndicatorError as e:
                        continue

                    # Intermediate data - indicator
                    if len(corporates) > 1:
                        try:
                            data_indicator = data.groupby('corporate').apply(lambda df: calculate_indicator_fundamental(df, indicator))
                        except IndicatorError as e:
                            # print(f'Calculation failed for {corporates} on indicator {indicator_name} from {start_year} to {end_year}: {e}')
                            continue
                        if not data_indicator.empty and data_indicator.index.nlevels == 3:
                            data_indicator = data_indicator.droplevel(0)
                    else:
                        try:
                            data_indicator = calculate_indicator_fundamental(data, indicator)
                        except IndicatorError as e:
                            # print(f'Calculation failed for {corporates} on indicator {indicator_name} from {start_year} to {end_year}: {e}')
                            continue
                    if isinstance(data_indicator, pd.Series):  # Convert to DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])

                    if len(data_indicator) == 0:  # No data after calculation, resample
                        continue

                    if (data_indicator == float('inf')).any().any() or (data_indicator == float('-inf')).any().any():
                        continue

                    try:
                        answer = function(data_indicator, **kwargs)
                    except Exception as e:
                        continue

                    answer = round(float(answer), 4)

                    valid_combination = True  # Validation passed, break while loop

                used_corporates.add(tuple(corporates))

                if 'AmortizationOfIntangibles' in indicator['data'] and (data['AmortizationOfIntangibles'] == 0).all():
                    data = data.drop(columns=['AmortizationOfIntangibles'])
                    data.rename(columns={'DepreciationExpense': 'Depreciation & Amortization'}, inplace=True)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f'Answer: {answer}')  ### debug
                else:
                    qa = {'question': question, 'answer': answer, 'difficulty': difficulty, 'sub-category': indicator['sub-category'], 'formula': indicator['formula'], 'data': data.to_csv()}
                    if difficulty != 'easy':
                        qa['data_indicator'] = data_indicator.to_csv()
                    QAs.append(qa)
    if QAs:
        pd.DataFrame(QAs).to_csv(output_path, index=False)
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(QAs, f, ensure_ascii=False, indent=4)


def generate_technical(output_path=None, question_template_technical=question_template_technical_ma, indicators_technical=indicators_technical_ma, add_note=True, n_per=5):
    var_pools = {
        'corporate': CORPORATE_LIST,
        'date': END_DATE_LIST,
        'end_date': END_DATE_LIST,
        't_day': T_DAY_LIST,
        'n': N_LIST,
        'n1': N_LIST,
        'n2': N_LIST,
        'x': X_LIST,
        'ohlc': OHLC_LIST,
        'compare': COMPARE_LIST,
        'breakthrough': BREAKTHROUGH_LIST,
        'ns': NS_LIST,
        'long': LONG_LIST,
        'channel': CHANNEL_LIST,
    }
    var_pools['indicator'] = list(indicators_technical.keys())

    QAs = []
    for template_group in question_template_technical:
        templates = template_group['templates']
        function = template_group['function']
        difficulty = template_group['difficulty']
        print(f"Generating QAs for template group: {templates[0]}")
        required_vars = [v[1] for v in string.Formatter().parse(templates[0]) if v[1] is not None]
        iter_vars = list(required_vars)

        if 'start_date' in required_vars and 'end_date' in required_vars:
            iter_vars[required_vars.index('start_date')] = 't_day'

        # Iterate over all indicators
        main_iter_var = 'indicator'
        iter_vars.remove(main_iter_var)
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_corporates = set()
            while len(used_corporates) < n_per:
                question_template = templates[len(used_corporates) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                # Use a while loop to ensure a valid parameter combination (e.g., n1 != n2)
                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    if 'n1' in kwargs and 'n2' in kwargs:
                        if kwargs['n1'] == kwargs['n2']:
                            continue  # Continue while loop to resample

                    # Calculate derived variables
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        kwargs['start_date'] = kwargs['end_date'] - datetime.timedelta(days=kwargs['t_day'])
                        kwargs.pop('t_day')

                    # Process date formats
                    for k in ['date', 'start_date', 'end_date']:
                        if k in kwargs:
                            kwargs[k] = kwargs[k].strftime('%B %d, %Y')

                    indicator = indicators_technical[kwargs[main_iter_var]]

                    # Get raw data
                    if 'date' in required_vars:
                        start_date = end_date = kwargs['date']
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        start_date = kwargs['start_date']
                        end_date = kwargs['end_date']

                    if 'n' in kwargs:
                        n_list = [kwargs['n']]
                    elif 'n1' in kwargs and 'n2' in kwargs:
                        n_list = [kwargs['n1'], kwargs['n2']]
                    elif 'ns' in kwargs:
                        n_list = list(map(int, kwargs['ns'].split(', ')))
                    else:
                        n_list = None
                    if n_list:
                        n_val = max(n_list)
                        parameters = {'n': n_val}
                    else:
                        # Get the maximum default value of integer parameters in the indicator formula
                        sig = inspect.signature(indicator['function'])
                        parameters = {}
                        for name, param in sig.parameters.items():
                            if param.default is not inspect.Parameter.empty and isinstance(param.default, int):
                                parameters[name] = param.default

                    if 'data_lookback_n' in indicator:
                        n_lookback = indicator['data_lookback_n'](**parameters)
                    else:
                        if len(parameters) > 0:
                            n_lookback = max(parameters.values()) - 1  # SMA only needs to look back n-1 days
                        else:
                            n_lookback = None

                    if 'ohlc' in kwargs:
                        ohlc = kwargs['ohlc']
                        if ohlc not in indicator['data']:
                            indicator['data'].append(ohlc)

                    # Validation: sampled corporate must not overlap with already-used ones
                    if kwargs['corporate'] in used_corporates:
                        continue

                    try:
                        data = get_data_df_technical(kwargs['corporate'], indicator, start_date, end_date, n_lookback)
                    except IndicatorError as e:
                        continue
                    if data is None:  # Non-trading day, resample
                        continue

                    if add_note:
                        note = f' ({indicator["note"]})' if 'note' in indicator else ''
                    else:
                        note = ''
                    question = question_template.format(**kwargs) + note + question_suffix_technical

                    # Intermediate data - indicator
                    data_indicator = calculate_indicator_technical(data, indicator, n_list)
                    if isinstance(data_indicator, pd.Series):  # Convert to DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])
                    if len(data_indicator) == 0:  # No data after calculation, resample
                        continue
                    if 'ohlc' in kwargs:
                        data_indicator = data_indicator.join(data[ohlc], how='left', on='date')
                    if 'breakthrough' in required_vars:
                        data_indicator = data_indicator.join(data['close'], how='left', on='date')

                    valid_combination = True  # Validation passed, break while loop

                used_corporates.add(kwargs['corporate'])

                answer = function(data_indicator, **kwargs)
                answer = round(float(answer), 4)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f"Answer: {answer}")
                else:
                    qa = {'question': question, 'answer': answer, 'difficulty': difficulty, 'sub-category': indicator['sub-category'], 'formula': indicator['formula'], 'data': data.to_csv()}
                    if difficulty != 'easy':
                        qa['data_indicator'] = data_indicator.to_csv()
                    QAs.append(qa)
    if QAs:
        pd.DataFrame(QAs).to_csv(output_path, index=False)
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(QAs, f, ensure_ascii=False, indent=4)


def generate_technical_performance(output_path=None, question_template_technical=question_template_technical_performance, indicators_technical=indicators_technical_performance, add_note=True, n_per=5):
    var_pools = {
        'corporate': CORPORATE_LIST,
        'end_date': END_DATE_LIST,
        't_day': T_DAY_LIST,
    }
    var_pools['indicator'] = list(indicators_technical.keys())

    QAs = []
    for template_group in question_template_technical:
        templates = template_group['templates']
        function = template_group['function']
        difficulty = template_group['difficulty']
        print(f"Generating QAs for template group: {templates[0]}")
        required_vars = [v[1] for v in string.Formatter().parse(templates[0]) if v[1] is not None]
        iter_vars = list(required_vars)

        if 'start_date' in required_vars and 'end_date' in required_vars:
            iter_vars[required_vars.index('start_date')] = 't_day'

        # Exhaust indicator
        main_iter_var = 'indicator'
        iter_vars.remove(main_iter_var)
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_corporates = set()
            while len(used_corporates) < n_per:
                question_template = templates[len(used_corporates) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    # Calculate derived variables
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        kwargs['start_date'] = kwargs['end_date'] - datetime.timedelta(days=kwargs['t_day'])
                        kwargs.pop('t_day')

                    # Process date formats
                    for k in ['date', 'start_date', 'end_date']:
                        if k in kwargs:
                            kwargs[k] = kwargs[k].strftime('%B %d, %Y')

                    indicator = indicators_technical[kwargs[main_iter_var]]

                    # Get raw data
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        start_date = kwargs['start_date']
                        end_date = kwargs['end_date']

                    # Validation: sampled corporate must not overlap with already-used ones
                    if kwargs['corporate'] in used_corporates:
                        continue

                    data = get_data_df_technical_performance(kwargs['corporate'], indicator, start_date, end_date)

                    if len(data) == 0:  # No data, resample
                        continue

                    if add_note:
                        note = f' ({indicator["note"]})' if 'note' in indicator else ''
                    else:
                        note = ''
                    question = question_template.format(**kwargs) + note + question_suffix_technical

                    # Intermediate data - indicator
                    data_indicator = calculate_indicator_technical(data, indicator)

                    valid_combination = True  # Validation passed, break while loop

                used_corporates.add(kwargs['corporate'])

                answer = function(data_indicator, **kwargs)
                answer = round(float(answer), 4)

                if indicator_name == 'Annualized Return':
                    data = data.iloc[[0,-1], :]

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f"Answer: {answer}")
                else:
                    qa = {'question': question, 'answer': answer, 'difficulty': difficulty, 'sub-category': indicator['sub-category'], 'formula': indicator['formula'], 'data': data.to_csv()}
                    if difficulty != 'easy':
                        qa['data_indicator'] = data_indicator.to_csv()
                    QAs.append(qa)
    if QAs:
        pd.DataFrame(QAs).to_csv(output_path, index=False)
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(QAs, f, ensure_ascii=False, indent=4)


def generate_macro(output_path=None, add_note=True, n_per=5):
    var_pools = {
        'year': END_YEAR_LIST,
        'end_year': END_YEAR_LIST,
        't_year': T_YEAR_LIST,
        'indicator': list(indicators_macro.keys()),
        'indicator1': list(indicators_macro.keys()),
        'statistic': STATISTIC_MACRO_LIST,
        'change': CHANGE_LIST,
    }

    QAs = []
    for template_group in question_template_macro:
        templates = template_group['templates']
        function = template_group['function']
        difficulty = template_group['difficulty']
        print(f"Generating QAs for template group: {templates[0]}")
        required_vars = [v[1] for v in string.Formatter().parse(templates[0]) if v[1] is not None]
        iter_vars = list(required_vars)

        if 'start_year' in required_vars and 'end_year' in required_vars:
            iter_vars[required_vars.index('start_year')] = 't_year'

        main_iter_var = 'indicator' if 'indicator' in iter_vars else 'indicator1'
        iter_vars.remove(main_iter_var)

        # Exhaust indicator1
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_periods = set()
            while len(used_periods) < n_per:
                question_template = templates[len(used_periods) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    # Calculate derived variables
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        kwargs['start_year'] = kwargs['end_year'] - kwargs['t_year']
                        kwargs.pop('t_year')

                    if add_note:
                        note = f' ({indicators_macro[indicator_name]["note"]})' if 'note' in indicators_macro[indicator_name] else ''
                    else:
                        note = ''

                    if 'How many times' in templates[0]:
                        note = note.replace('Unit:', 'Indicator unit:')

                    question = question_template.format(**kwargs) + note + question_suffix_macro

                    indicator = indicators_macro[kwargs[main_iter_var]]

                    # Get raw data
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        start_year = kwargs['start_year']
                        end_year = kwargs['end_year']
                    if 'year' in required_vars:
                        start_year = end_year = kwargs['year']

                    # Validation: year range must not have been used before
                    period_key = (start_year, end_year)
                    if period_key in used_periods:
                        continue

                    try:
                        data = get_data_df_macro(indicator, start_year, end_year, indicator_name)
                    except IndicatorError as e:
                        continue

                    # Intermediate data - indicator
                    data_indicator = calculate_indicator_macro(data, indicator, indicator_name)

                    if isinstance(data_indicator, pd.Series):  # Convert to DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])

                    answer = function(data_indicator, **kwargs)
                    answer = round(float(answer), 4)

                    valid_combination = True  # Validation passed, break while loop

                used_periods.add(period_key)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f'Answer: {answer}')  ### debug
                else:
                    qa = {'question': question, 'answer': answer, 'difficulty': difficulty, 'sub-category': indicator['sub-category'], 'formula': indicator['formula'], 'data': data.to_csv()}
                    if difficulty != 'easy':
                        qa['data_indicator'] = data_indicator.to_csv()
                    QAs.append(qa)
    if QAs:
        pd.DataFrame(QAs).to_csv(output_path, index=False)



def generate_macro_tmp(output_path=None, add_note=True, n_per=5):
    var_pools = {
        'year': END_YEAR_LIST,
        'end_year': END_YEAR_LIST,
        't_year': T_YEAR_LIST,
        'indicator': list(indicators_macro.keys()),
        'indicator1': list(indicators_macro.keys()),
        'statistic': STATISTIC_MACRO_LIST,
        'change': CHANGE_LIST,
    }

    QAs = []
    for template_group in question_template_macro_tmp:
        templates = template_group['templates']
        function = template_group['function']
        difficulty = template_group['difficulty']
        print(f"Generating QAs for template group: {templates[0]}")
        required_vars = [v[1] for v in string.Formatter().parse(templates[0]) if v[1] is not None]
        iter_vars = list(required_vars)

        if 'start_year' in required_vars and 'end_year' in required_vars:
            iter_vars[required_vars.index('start_year')] = 't_year'

        main_iter_var = 'indicator' if 'indicator' in iter_vars else 'indicator1'
        iter_vars.remove(main_iter_var)

        # Exhaust indicator1
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_periods = set()
            while len(used_periods) < n_per:
                question_template = templates[len(used_periods) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    # Calculate derived variables
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        kwargs['start_year'] = kwargs['end_year'] - kwargs['t_year']
                        kwargs.pop('t_year')

                    if add_note:
                        note = f' ({indicators_macro[indicator_name]["note"]})' if 'note' in indicators_macro[indicator_name] else ''
                    else:
                        note = ''

                    if 'How many times' in templates[0]:
                        note = note.replace('Unit:', 'Indicator unit:')

                    question = question_template.format(**kwargs) + note + question_suffix_macro

                    indicator = indicators_macro[kwargs[main_iter_var]]

                    # Get raw data
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        start_year = kwargs['start_year']
                        end_year = kwargs['end_year']
                    if 'year' in required_vars:
                        start_year = end_year = kwargs['year']

                    # Validation: year range must not have been used before
                    period_key = (start_year, end_year)
                    if period_key in used_periods:
                        continue

                    try:
                        data = get_data_df_macro(indicator, start_year, end_year, indicator_name)
                    except IndicatorError as e:
                        continue

                    # Intermediate data - indicator
                    data_indicator = calculate_indicator_macro(data, indicator, indicator_name)

                    if isinstance(data_indicator, pd.Series):  # Convert to DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])

                    answer = function(data_indicator, **kwargs)
                    answer = round(float(answer), 4)

                    valid_combination = True  # Validation passed, break while loop

                used_periods.add(period_key)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f'Answer: {answer}')  ### debug
                else:
                    qa = {'question': question, 'answer': answer, 'difficulty': difficulty, 'sub-category': indicator['sub-category'], 'formula': indicator['formula'], 'data': data.to_csv()}
                    if difficulty != 'easy':
                        qa['data_indicator'] = data_indicator.to_csv()
                    QAs.append(qa)
    if QAs:
        pd.DataFrame(QAs).to_csv(output_path, index=False)



if __name__ == '__main__':
    output_dir = 'QA'
    n_per = 5
    generate_fundamental(f'{output_dir}/fundamental.csv', n_per=n_per)
    generate_technical(f'{output_dir}/technical_ma.csv', question_template_technical_ma, indicators_technical_ma, n_per=n_per)
    generate_technical(f'{output_dir}/technical_channel.csv', question_template_technical_channel, indicators_technical_channel, n_per=n_per)
    generate_technical(f'{output_dir}/technical_n.csv', question_template_technical_other, indicators_technical_n, n_per=n_per)
    generate_technical(f'{output_dir}/technical_other.csv', question_template_technical_other, indicators_technical_other, n_per=n_per)
    generate_technical_performance(f'{output_dir}/technical_performance.csv', question_template_technical_performance, indicators_technical_performance, n_per=n_per)
    generate_macro(f'{output_dir}/macro.csv', n_per=n_per)
