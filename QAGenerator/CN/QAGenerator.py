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
    if df.isna().mean().mean() >= 0.4:  # 如果nan值太多，触发重新抽样
        raise IndicatorError('Too much missing data')
    for col in ['利润总额']:
        if col in df.columns and not df[col].fillna(0).ne(0).all():
            raise IndicatorError(f'{col} must be non-zero')
    for col in ['财务费用——利息费用', '应交税费', '长期应付款合计', '非流动负债合计']:
        if col in df.columns and not df[col].fillna(0).ge(0).all():
            raise IndicatorError(f'{col} must be non-negative')

def get_one_data_df_fundamental(corporate, indicator, start_year, end_year):
    df = pd.read_csv(f'./data/{corporate}/fundamental.csv')

    # 有 shift(1) 的指标需要前一年数据，因此 start_year 需要减 1
    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function'])
    ):
        start_year -= 1
    
    if start_year < 2016:
        raise IndicatorError(f'Start year {start_year} too early!')

    df = df.query(
        f'metric_name in {indicator["data"]} and '
        f'report_period in {[f"{year}-4" for year in range(start_year, end_year+1)]}'
    )
    df['year'] = df['report_period'].str[:4]
    df = df.pivot_table(index='year', columns='metric_name', values='value').sort_index()
    df = df.reindex(columns=indicator["data"])

    if len(df) != (end_year - start_year + 1):
        raise IndicatorError(f'Not enough data for {corporate} from {start_year} to {end_year}')

    check_df_fundamental(df)
    df = df.fillna(0)
    return df

def get_data_df_fundamental(corporates, indicator, start_year, end_year):
    if len(corporates) == 1:
        return get_one_data_df_fundamental(corporates[0], indicator, start_year, end_year)
    else:
        for corporate in corporates:
            for keyword in ['证券', '银行', '保险', '人保', '中金']:
                if keyword in corporate:
                    raise IndicatorError('Securities companies are not supported for multi-corporate questions')
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

    start_date = pd.to_datetime(start_date, format='%Y年%m月%d日').date()
    end_date = pd.to_datetime(end_date, format='%Y年%m月%d日').date()

    if start_date == end_date:  # 单点计算时要判断是否为非交易日
        if df.query(f'date == "{start_date}"').empty:
            return None  # 非交易日

    if n is not None:  # starte_date 要往前推 n 个交易日
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
        df_n = df_n.to_frame(name=f"{n}日")
        df_list.append(df_n)
    return pd.concat(df_list, axis=1).dropna()

def get_data_df_technical_performance(corporate, indicator, start_date, end_date):
    df = pd.read_csv(f'./data/{corporate}/technical.csv', parse_dates=['date'])

    start_date = pd.to_datetime(start_date, format='%Y年%m月%d日').date()
    end_date = pd.to_datetime(end_date, format='%Y年%m月%d日').date()

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
        df_index = pd.read_csv(f'data/CSI300.csv', parse_dates=['date']).set_index('date').sort_index()[['close']]
        df_index = df_index.query(f'date >= "{start_date}" and date <= "{end_date}"')
        if 'benchmark_close' in indicator['data']:
            df['benchmark_close'] = df_index
        if 'market_close' in indicator['data']:
            df['market_close'] = df_index
    return df

def get_data_df_macro(indicator, start_year, end_year, indicator_name=None):
    df = pd.read_csv(f'./data/macro.csv')

    # 有 shift(1) 的指标需要前一年数据，因此 start_year 需要减 1
    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function'])
    ):
        start_year -= 1

    if start_year < 2016:
        raise IndicatorError(f'Start year {start_year} too early!')

    if indicator_name in ['GDP平减指数', '劳动生产率']:
        df = df.query(f'(year >= {start_year} and year <= {end_year}) or year == 1978')
    else:
        df = df.query(f'year >= {start_year} and year <= {end_year}')
    df = df.set_index('year').sort_index()
    df = df.reindex(columns=indicator["data"])

    if df.isna().sum().sum() > 0:
        raise IndicatorError('Missing data for macro indicator')

    return df

def calculate_indicator_macro(df: pd.DataFrame, indicator: dict, indicator_name=None):
    df = indicator['function'](df)
    # 有 shift(1) 的指标需要去掉第一行
    if (
        '.shift' in inspect.getsource(indicator['function']) or
        '.pct_change' in inspect.getsource(indicator['function']) or
        '.diff' in inspect.getsource(indicator['function']) or
        indicator_name in ['GDP平减指数', '劳动生产率']
    ):
        df = df.iloc[1:]
    return df



def generate_fundamental(output_path=None, add_note=True, n_per=5):
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

        # 确定主循环变量（优先找 indicator，若没有就找 indicator1）
        main_iter_var = 'indicator' if 'indicator' in iter_vars else 'indicator1'
        if main_iter_var in iter_vars:
            iter_vars.remove(main_iter_var)

        # 穷举 indicator
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_corporates = set()
            while len(used_corporates) < n_per:
                question_template = templates[len(used_corporates) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                # 由于可能出现 corporate1==corporate2，我们用一个 while 循环确保拿到合法的参数组合
                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    if 'corporates' in required_vars:
                        kwargs['corporates'] = '、'.join(random.sample(CORPORATE_LIST, random.randint(3, 5)))
                        # kwargs['corporates'] = '、'.join(random.sample(CORPORATE_LIST, random.randint(2, 2)))

                    # 校验：去掉 corporate1 == corporate2 的非法情况
                    if 'corporate1' in kwargs and 'corporate2' in kwargs:
                        if kwargs['corporate1'] == kwargs['corporate2']:
                            continue  # 继续 while 循环重新抽样

                    # 校验：statistic == '复合年均增长率' 时，indicator 只能是部分特定指标
                    if 'statistic' in kwargs and kwargs['statistic'] == '复合年均增长率':
                        if kwargs[main_iter_var] not in ['资产负债率', '产权比率', 'EBIT', 'EBITDA', 'NOPAT', '每股净资产', '每股资本公积金', '每股未分配利润', '每股经营现金流']:
                            continue  # 继续 while 循环重新抽样

                    # 计算衍生变量
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        kwargs['start_year'] = kwargs['end_year'] - kwargs['t_year']
                        kwargs.pop('t_year')

                    indicator = indicators_fundamental[kwargs[main_iter_var]]

                    if add_note:
                        note = f'（{indicator["note"]}）' if 'note' in indicator else ''
                    else:
                        note = ''

                    if '之比是多少' in templates[0] or '有几次' in templates[0]:
                        note = note.replace('单位：', '指标单位：')

                    question = question_template.format(**kwargs) + note + question_suffix_fundamental
                    # print(question)  ### debug

                    # 获取原始数据
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        start_year = kwargs['start_year']
                        end_year = kwargs['end_year']
                    if 'year' in required_vars:
                        start_year = end_year = kwargs['year']

                    if 'corporates' in required_vars:
                        corporates = kwargs['corporates'].split('、')
                    elif 'corporate' in required_vars:
                        corporates = [kwargs['corporate']]
                    elif 'corporate1' in required_vars and 'corporate2' in required_vars:
                        corporates = [kwargs['corporate1'], kwargs['corporate2']]
                    else:
                        raise IndicatorError('No corporate variable found in question template.')

                    # 校验：此次采样的公司与已用公司不重叠
                    if {tuple(corporates)} & used_corporates:
                        continue

                    try:
                        data = get_data_df_fundamental(corporates, indicator, start_year, end_year)
                    except IndicatorError as e:
                        continue

                    # 中间数据-指标
                    if len(corporates) > 1:
                        try:
                            data_indicator = data.groupby('corporate').apply(lambda df: calculate_indicator_fundamental(df, indicator))
                        except IndicatorError as e:
                            continue
                        if not data_indicator.empty and data_indicator.index.nlevels == 3:
                            data_indicator = data_indicator.droplevel(0)
                    else:
                        try:
                            data_indicator = calculate_indicator_fundamental(data, indicator)
                        except IndicatorError as e:
                            continue
                    if isinstance(data_indicator, pd.Series):  # 转成 DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])

                    if len(data_indicator) == 0:  # 计算后没有数据，重新抽样
                        continue

                    if (data_indicator == float('inf')).any().any() or (data_indicator == float('-inf')).any().any():
                        continue

                    try:
                        answer = function(data_indicator, **kwargs)
                    except IndicatorError as e:
                        continue

                    answer = round(float(answer), 4)

                    valid_combination = True  # 校验通过，打破 while

                used_corporates.add(tuple(corporates))

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f'答案：{answer}')  ### debug
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

        # 穷举 indicator
        main_iter_var = 'indicator'
        iter_vars.remove(main_iter_var)
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_corporates = set()
            while len(used_corporates) < n_per:
                question_template = templates[len(used_corporates) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                # 可能出现 n1==n2 的非法情况，我们用一个 while 循环确保拿到合法的参数组合
                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    if 'n1' in kwargs and 'n2' in kwargs:
                        if kwargs['n1'] == kwargs['n2']:
                            continue  # 继续 while 循环重新抽样

                    # 计算衍生变量
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        kwargs['start_date'] = kwargs['end_date'] - datetime.timedelta(days=kwargs['t_day'])
                        kwargs.pop('t_day')

                    # 处理日期格式
                    for k in ['date', 'start_date', 'end_date']:
                        if k in kwargs:
                            kwargs[k] = kwargs[k].strftime('%Y年%m月%d日')

                    indicator = indicators_technical[kwargs[main_iter_var]]

                    # 获取原始数据
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
                        n_list = list(map(int, kwargs['ns'].split('、')))
                    else:
                        n_list = None
                    if n_list:
                        n_val = max(n_list)
                        parameters = {'n': n_val}
                    else:
                        # 获取指标公式中整数型参数默认值的最大值
                        sig = inspect.signature(indicator['function'])
                        parameters = {}
                        for name, param in sig.parameters.items():
                            if param.default is not inspect.Parameter.empty and isinstance(param.default, int):
                                parameters[name] = param.default

                    if 'data_lookback_n' in indicator:
                        n_lookback = indicator['data_lookback_n'](**parameters)
                    else:
                        if len(parameters) > 0:
                            n_lookback = max(parameters.values()) - 1  # SMA 只需往前推 n-1 天
                        else:
                            n_lookback = None

                    if 'ohlc' in kwargs:
                        ohlc_dict = {
                            '开盘价': 'open',
                            '最高价': 'high',
                            '最低价': 'low',
                            '收盘价': 'close',
                        }
                        ohlc = ohlc_dict[kwargs['ohlc']]
                        if ohlc not in indicator['data']:
                            indicator['data'].append(ohlc)

                    # 校验：此次采样的公司与已用公司不重叠
                    if kwargs['corporate'] in used_corporates:
                        continue

                    try:
                        data = get_data_df_technical(kwargs['corporate'], indicator, start_date, end_date, n_lookback)
                    except IndicatorError as e:
                        continue
                    if data is None:  # 非交易日，重新抽样
                        continue

                    if add_note:
                        note = f'（{indicator["note"]}）' if 'note' in indicator else ''
                    else:
                        note = ''
                    question = question_template.format(**kwargs) + note + question_suffix_technical

                    # 中间数据-指标
                    data_indicator = calculate_indicator_technical(data, indicator, n_list)
                    if isinstance(data_indicator, pd.Series):  # 转成 DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])
                    if len(data_indicator) == 0:  # 计算后没有数据，重新抽样
                        continue
                    if 'ohlc' in kwargs:
                        data_indicator = data_indicator.join(data[ohlc], how='left', on='date')
                    if '价格' in templates[0]:
                        data_indicator = data_indicator.join(data['close'], how='left', on='date')

                    valid_combination = True  # 校验通过，打破 while

                used_corporates.add(kwargs['corporate'])

                answer = function(data_indicator, **kwargs)
                answer = round(float(answer), 4)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f"答案：{answer}")
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

        # 穷举 indicator
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

                    # 计算衍生变量
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        kwargs['start_date'] = kwargs['end_date'] - datetime.timedelta(days=kwargs['t_day'])
                        kwargs.pop('t_day')

                    # 处理日期格式
                    for k in ['date', 'start_date', 'end_date']:
                        if k in kwargs:
                            kwargs[k] = kwargs[k].strftime('%Y年%m月%d日')

                    indicator = indicators_technical[kwargs[main_iter_var]]

                    # 获取原始数据
                    if 'start_date' in required_vars and 'end_date' in required_vars:
                        start_date = kwargs['start_date']
                        end_date = kwargs['end_date']

                    # 校验：此次采样的公司与已用公司不重叠
                    if kwargs['corporate'] in used_corporates:
                        continue

                    data = get_data_df_technical_performance(kwargs['corporate'], indicator, start_date, end_date)

                    if len(data) == 0:  # 没有数据，重新抽样
                        continue

                    if add_note:
                        note = f'（{indicator["note"]}）' if 'note' in indicator else ''
                    else:
                        note = ''

                    question = question_template.format(**kwargs) + note + question_suffix_technical

                    # 中间数据-指标
                    data_indicator = calculate_indicator_technical(data, indicator)

                    valid_combination = True  # 校验通过，打破 while

                used_corporates.add(kwargs['corporate'])

                answer = function(data_indicator, **kwargs)
                answer = round(float(answer), 4)

                if indicator_name == '年化收益率':
                    data = data.iloc[[0,-1], :]

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f"答案：{answer}")
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

        # 穷举 indicator1
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_periods = set()
            while len(used_periods) < n_per:
                question_template = templates[len(used_periods) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    # 计算衍生变量
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        kwargs['start_year'] = kwargs['end_year'] - kwargs['t_year']
                        kwargs.pop('t_year')

                    if add_note:
                        note = f'（{indicators_macro[indicator_name]["note"]}）' if 'note' in indicators_macro[indicator_name] else ''
                    else:
                        note = ''

                    if '有几次' in templates[0]:
                        note = note.replace('单位：', '指标单位：')

                    question = question_template.format(**kwargs) + note + question_suffix_macro

                    indicator = indicators_macro[kwargs[main_iter_var]]

                    # 获取原始数据
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        start_year = kwargs['start_year']
                        end_year = kwargs['end_year']
                    if 'year' in required_vars:
                        start_year = end_year = kwargs['year']

                    # 校验：年份（区间）未被使用过
                    period_key = (start_year, end_year)
                    if period_key in used_periods:
                        continue

                    try:
                        data = get_data_df_macro(indicator, start_year, end_year, indicator_name)
                    except IndicatorError as e:
                        continue

                    # 中间数据-指标
                    data_indicator = calculate_indicator_macro(data, indicator, indicator_name)
                    if isinstance(data_indicator, pd.Series):  # 转成 DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])

                    answer = function(data_indicator, **kwargs)
                    answer = round(float(answer), 4)

                    valid_combination = True  # 校验通过，打破 while

                used_periods.add(period_key)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f'答案：{answer}')  ### debug
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

        # 穷举 indicator1
        for indicator_name in tqdm(var_pools[main_iter_var]):
            used_periods = set()
            while len(used_periods) < n_per:
                question_template = templates[len(used_periods) % len(templates)]
                kwargs = {main_iter_var: indicator_name}

                valid_combination = False
                while not valid_combination:
                    for var in iter_vars:
                        kwargs[var] = random.choice(var_pools[var])

                    # 计算衍生变量
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        kwargs['start_year'] = kwargs['end_year'] - kwargs['t_year']
                        kwargs.pop('t_year')

                    if add_note:
                        note = f'（{indicators_macro[indicator_name]["note"]}）' if 'note' in indicators_macro[indicator_name] else ''
                    else:
                        note = ''

                    if '有几次' in templates[0]:
                        note = note.replace('单位：', '指标单位：')

                    question = question_template.format(**kwargs) + note + question_suffix_macro

                    indicator = indicators_macro[kwargs[main_iter_var]]

                    # 获取原始数据
                    if 'start_year' in required_vars and 'end_year' in required_vars:
                        start_year = kwargs['start_year']
                        end_year = kwargs['end_year']
                    if 'year' in required_vars:
                        start_year = end_year = kwargs['year']

                    # 校验：年份（区间）未被使用过
                    period_key = (start_year, end_year)
                    if period_key in used_periods:
                        continue

                    try:
                        data = get_data_df_macro(indicator, start_year, end_year, indicator_name)
                    except IndicatorError as e:
                        continue

                    # 中间数据-指标
                    data_indicator = calculate_indicator_macro(data, indicator, indicator_name)
                    if isinstance(data_indicator, pd.Series):  # 转成 DataFrame
                        data_indicator = data_indicator.to_frame(name=kwargs[main_iter_var])

                    answer = function(data_indicator, **kwargs)
                    answer = round(float(answer), 4)

                    valid_combination = True  # 校验通过，打破 while

                used_periods.add(period_key)

                if output_path is None:
                    print(question)  ### debug
                    print(data)  ### debug
                    print(data_indicator)  ### debug
                    print(f'答案：{answer}')  ### debug
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
