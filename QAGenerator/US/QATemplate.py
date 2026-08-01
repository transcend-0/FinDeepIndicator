import datetime
import pandas as pd
from QACalculator import IndicatorError



def calculate_statistic(df: pd.DataFrame, indicator: str, statistic: str, **kwargs):
    df_indicator_col = df[indicator]
    if statistic == 'average':
        return df_indicator_col.mean()
    elif statistic == 'maximum':
        return df_indicator_col.max()
    elif statistic == 'minimum':
        return df_indicator_col.min()
    elif statistic == 'CAGR':
        start_value = df_indicator_col.iloc[0]
        end_value = df_indicator_col.iloc[-1]
        n = len(df_indicator_col) - 1
        if start_value <= 0 or end_value <= 0:
            raise IndicatorError("CAGR is not defined for non-positive values.")
        return ((end_value / start_value) ** (1/n) - 1) * 100
    elif statistic in ['YoY growth rate', 'QoQ growth rate']:
        start_value = df_indicator_col.iloc[0]
        end_value = df_indicator_col.iloc[-1]
        if start_value <= 0:
            raise IndicatorError("Growth rate is not defined for non-positive values.")
        return (end_value - start_value) / start_value * 100
    else:
        raise IndicatorError(f"Unsupported statistic: {statistic}")

def calculate_statistic_and_compare(df: pd.DataFrame, statistic: str, indicator1: str, corporate1: str, corporate2: str, **kwargs):
    stat1 = calculate_statistic(df.query(f'corporate == "{corporate1}"'), indicator1, statistic)
    stat2 = calculate_statistic(df.query(f'corporate == "{corporate2}"'), indicator1, statistic)
    if stat2 == 0:
        raise IndicatorError('Cannot compare when the second statistic is zero.')
    return stat1 / stat2

def indicator_and_compare(df: pd.DataFrame, indicator: str, high_or_low: str, **kwargs):
    if high_or_low == 'high':
        best_corp = df.idxmax()
        best_value = df.max()
        second_value = df[indicator].nlargest(2).iloc[-1]
        return best_value - second_value
    elif high_or_low == 'low':
        best_corp = df.idxmin()
        best_value = df.min()
        second_value = df[indicator].nsmallest(2).iloc[-1]
        return second_value - best_value
    else:
        raise IndicatorError(f"Unsupported high_or_low: {high_or_low}")

def indicator_another(df: pd.DataFrame, indicator1: str, high_or_low: str, indicator2: str, **kwargs):
    if high_or_low == 'high':
        best_year = df[indicator1].idxmax()
    elif high_or_low == 'low':
        best_year = df[indicator1].idxmin()
    else:
        raise IndicatorError(f"Unsupported high_or_low: {high_or_low}")
    return df.loc[best_year, indicator2]

def count_change(df: pd.DataFrame, indicator1: str, change: str, **kwargs):
    count = 0
    for i in range(1, len(df)):
        if change == 'increase' and df[indicator1].iloc[i] > df[indicator1].iloc[i-1]:
            count += 1
        elif change == 'decrease' and df[indicator1].iloc[i] < df[indicator1].iloc[i-1]:
            count += 1
    return count


question_template_fundamental = [
    # 'What is the {indicator} of {corporate} in {year} {quarter}?',  # single-point calculation
    # 'What is the {statistic_quarter} of {indicator1} for {corporate} in {year} {quarter}?',  # point-in-time statistics
    {
        'templates': [
            'What is the {indicator} of {corporate} in {year}?',
            'What was {corporate}\'s {indicator} in {year}?',
            'In {year}, what was the {indicator} reported by {corporate}?',
            'For {corporate}, what is the value of {indicator} in {year}?',
            'Calculate the {indicator} for {corporate} in {year}.',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # single-point calculation
    },
    {
        'templates': [
            'What is the {statistic} of {indicator} for {corporate} from {start_year} to {end_year}?',
            'From {start_year} to {end_year}, what is the {statistic} of {corporate}\'s {indicator}?',
            'What was the {statistic} of {indicator} for {corporate} during {start_year}-{end_year}?',
            'For {corporate}\'s {indicator}, what is the {statistic} over {start_year} to {end_year}?',
            'Calculate the {statistic} of {corporate}\'s {indicator} from {start_year} to {end_year}.',
        ],
        'difficulty': 'medium',
        'function': calculate_statistic,  # period statistics
    },
    {
        'templates': [
            'Comparing the {statistic} of {indicator1} from {start_year} to {end_year}, what is the ratio of {corporate1} to {corporate2}?',
            'What is the ratio of {corporate1} to {corporate2} in terms of the {statistic} of {indicator1} from {start_year} to {end_year}?',
            'From {start_year} to {end_year}, how does the {statistic} of {indicator1} for {corporate1} compare to {corporate2} as a ratio?',
            'Over {start_year}-{end_year}, what is the ratio of {corporate1} to {corporate2} for the {statistic} of {indicator1}?',
            'Calculate the {statistic} of {indicator1} for {corporate1} and {corporate2} separately from {start_year} to {end_year}, and compute the ratio of {corporate1} to {corporate2}.',
        ],
        'difficulty': 'hard',
        'function': calculate_statistic_and_compare,  # cross-company comparison
    },
    {
        'templates': [
            'Among {corporates}, how much {high_or_low}er is the {indicator} for the top company compared to the second place in {year}?',
            'Among {corporates}, what is the gap in {indicator} between the {high_or_low}est-ranked company and the runner-up in {year}?',
            'In {year}, within {corporates}, by how much does the company with the {high_or_low}est {indicator} differ from the second-ranked company?',
            'For {indicator} in {year}, what is the difference between the top and second-place companies among {corporates}?',
            'Among {corporates}, calculate the margin between the {high_or_low}est-{indicator} company and the second-place company in {year}.',
        ],
        'difficulty': 'hard',
        'function': indicator_and_compare,  # cross-company comparison
    },
    # 'From {start_year} to {end_year}, in the year when {corporate}\'s {indicator1} was {high_or_low}est, what was {indicator2}?': indicator_another,  # extreme value location + derived calculation
    {
        'templates': [
            'Calculate the {indicator1} of {corporate} from {start_year} to {end_year} and make a year-over-year comparison. How many times was there a YoY {change}?',
            'Within the {start_year}–{end_year} window, how many times did {corporate}\'s {indicator1} {change} from one year to the next?',
            'Comparing {corporate}\'s {indicator1} year by year from {start_year} to {end_year}, how many times did it {change} relative to the preceding year?',
            'Taking {start_year} as the starting reference, how many year-over-year {change}s occurred in {corporate}\'s {indicator1} up to {end_year}?',
            'For {corporate}\'s {indicator1} from {start_year} to {end_year}, count the number of YoY {change}s between each consecutive pair of years.',
        ],
        'difficulty': 'hard',
        'function': count_change,  # trend change count
    },
]

question_suffix_fundamental = ' According to US GAAP, write down the indicator formula, collect the raw data, calculate the answer, and keep four decimal places. Please list the raw data and key calculation process.'

df = pd.read_csv('data/sp500_list_2025-12.csv')
CORPORATE_LIST = df['code_name'].tolist()

QUARTER_LIST = ['Q1', 'Q2', 'Q3', 'Q4']

END_YEAR_LIST = list(range(2016, 2026))
T_YEAR_LIST = [2, 4]

STATISTIC_QUARTER_LIST = ['YoY growth rate', 'QoQ growth rate']
STATISTIC_LIST = ['average', 'maximum', 'minimum', 'CAGR']
UNIT1_LIST = ['%']
UNIT2_LIST = ['Billion USD', 'USD']

HIGH_OR_LOW_LIST = ['high', 'low']
CHANGE_LIST = ['increase', 'decrease']



def technical_ma_count_x(df: pd.DataFrame, n1: int, n2: int, x: str, **kwargs):
    n_fast, n_slow = min(n1, n2), max(n1, n2)
    count = 0
    if x == 'golden cross':
        for i in range(1, len(df)):
            if df[f'{n_fast}-day'].iloc[i] > df[f'{n_slow}-day'].iloc[i] and df[f'{n_fast}-day'].iloc[i-1] <= df[f'{n_slow}-day'].iloc[i-1]:
                count += 1
    elif x == 'death cross':
        for i in range(1, len(df)):
            if df[f'{n_fast}-day'].iloc[i] < df[f'{n_slow}-day'].iloc[i] and df[f'{n_fast}-day'].iloc[i-1] >= df[f'{n_slow}-day'].iloc[i-1]:
                count += 1
    return count

def technical_ma_account_compare(df: pd.DataFrame, ohlc: str, compare: str, n: int, **kwargs):
    count = 0
    for i in range(len(df)):
        if compare == 'greater' and df[f'{n}-day'].iloc[i] > df[ohlc].iloc[i]:
            count += 1
        elif compare == 'less' and df[f'{n}-day'].iloc[i] < df[ohlc].iloc[i]:
            count += 1
    return count / len(df) * 100

def technical_ma_count_breakthrough(df: pd.DataFrame, breakthrough: str, n: int, **kwargs):
    count = 0
    for i in range(1, len(df)):
        if breakthrough == 'break above' and df['close'].iloc[i] > df[f'{n}-day'].iloc[i] and df['close'].iloc[i-1] < df[f'{n}-day'].iloc[i-1]:
            count += 1
        elif breakthrough == 'break below' and df['close'].iloc[i] < df[f'{n}-day'].iloc[i] and df['close'].iloc[i-1] > df[f'{n}-day'].iloc[i-1]:
            count += 1
    return count

def technical_ma_account_long(df: pd.DataFrame, ns: list, long: str, **kwargs):
    n_list = sorted(list(map(int, ns.split(', '))))
    count = 0
    for i in range(len(df)):
        if long == 'long' and all(df[f'{n}-day'].iloc[i] > df[f'{m}-day'].iloc[i] for n, m in zip(n_list[:-1], n_list[1:])):
            count += 1
        elif long == 'short' and all(df[f'{n}-day'].iloc[i] < df[f'{m}-day'].iloc[i] for n, m in zip(n_list[:-1], n_list[1:])):
            count += 1
    return count / len(df) * 100

question_template_technical_ma = [
    {
        'templates': [
            'What is the {n}-day period {indicator} close price of {corporate} on {date}? (Unit: USD)',
            'On {date}, what is the {n}-day {indicator} close price of {corporate}? (Unit: USD)',
            'For the {n}-day {indicator} close price, what is the value for {corporate} on {date}? (Unit: USD)',
            'What is {corporate}\'s {n}-day {indicator} close price on {date}? (Unit: USD)',
            'Calculate the {n}-day period {indicator} close price for {corporate} on {date}. (Unit: USD)',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # single-point calculation
    },
    # 'What is the log return of the {n}-day period {indicator} for {corporate} on {date}?',  # single-point calculation
    {
        'templates': [
            'From {start_date} to {end_date}, how many times did a {x} occur between the {n1}-day and {n2}-day {indicator} lines for {corporate}?',
            'How many {x} occurred between the {n1}-day and {n2}-day {indicator} lines for {corporate} from {start_date} to {end_date}?',
            'During {start_date} to {end_date}, how many times did {corporate}\'s {n1}-day and {n2}-day {indicator} lines form a {x}?',
            'For {corporate}, how many times did the {n1}-day and {n2}-day {indicator} lines produce a {x} between {start_date} and {end_date}?',
            'From {start_date} to {end_date}, count the number of {x} occurrences between the {n1}-day and {n2}-day {indicator} lines for {corporate}.',
        ],
        'difficulty': 'medium',
        'function': technical_ma_count_x,  # pattern: golden/death cross
    },
    {
        'templates': [
            'What is the percentage of days where the {ohlc} price is {compare} than the {n}-day {indicator} line for {corporate} from {start_date} to {end_date}? (Unit: %)',
            'From {start_date} to {end_date}, what proportion of days had {corporate}\'s {ohlc} price {compare} than its {n}-day {indicator} line? (Unit: %)',
            'During {start_date} to {end_date}, what percentage of trading days had {corporate}\'s {ohlc} {compare} the {n}-day {indicator} line?',
            'For {corporate} from {start_date} to {end_date}, what percentage of days had the {ohlc} {compare} than the {n}-day {indicator} line? (Unit: %)',
            'Calculate the percentage of days where {corporate}\'s {ohlc} was {compare} than its {n}-day {indicator} line from {start_date} to {end_date}. (Unit: %)',
        ],
        'difficulty': 'medium',
        'function': technical_ma_account_compare,  # pattern: long/short ratio
    },
    {
        'templates': [
            'How many times did the price {breakthrough} the {n}-day {indicator} line for {corporate} from {start_date} to {end_date}?',
            'From {start_date} to {end_date}, how many times did {corporate}\'s price {breakthrough} the {n}-day {indicator} line?',
            'For {corporate} between {start_date} and {end_date}, how many times did the price {breakthrough} the {n}-day {indicator} line?',
            'How many {breakthrough} events of the {n}-day {indicator} line occurred for {corporate} from {start_date} to {end_date}?',
            'During {start_date} to {end_date}, count the number of times {corporate}\'s close price {breakthrough} the {n}-day {indicator} line.',
        ],
        'difficulty': 'medium',
        'function': technical_ma_count_breakthrough,  # pattern: break above/below
    },
    {
        'templates': [
            'What is the percentage of days with {long} alignment for the {ns}-day {indicator} lines for {corporate} from {start_date} to {end_date}? (Unit: %)',
            'From {start_date} to {end_date}, what proportion of days had the {ns}-day {indicator} lines of {corporate} in {long} alignment? (Unit: %)',
            'During {start_date} to {end_date}, what percentage of trading days showed {long} alignment of {corporate}\'s {ns}-day {indicator} lines?',
            'For {corporate} from {start_date} to {end_date}, what percentage of days were the {ns}-day {indicator} lines in {long} alignment? (Unit: %)',
            'Calculate the percentage of days with {long} alignment in the {ns}-day {indicator} lines for {corporate} from {start_date} to {end_date}.',
        ],
        'difficulty': 'hard',
        'function': technical_ma_account_long,  # pattern: long/short alignment
    },
]

question_suffix_technical = ' Using unadjusted prices, write down the indicator formula, collect the raw data, calculate the answer, and keep four decimal places. Please list the raw data and key calculation process.'

END_DATE_LIST = pd.date_range(start='2020-01-01', end='2025-12-31').tolist()
T_DAY_LIST = list(range(10, 366))
# start_date = end_date - datetime.timedelta(days=t_day)

X_LIST = ['golden cross', 'death cross']
N_LIST = [5, 10, 20, 30, 60]
OHLC_LIST = ['open', 'high', 'low', 'close']
COMPARE_LIST = ['greater', 'less']
BREAKTHROUGH_LIST = ['break above', 'break below']
NS_LIST = ['5, 10, 20', '10, 20, 30', '5, 20, 60']
LONG_LIST = ['long', 'short']


def technical_channel_value(df: pd.DataFrame, channel: str, **kwargs):
    channel_dict = {
        'upper': 'Upper',
        'middle': 'Middle',
        'lower': 'Lower',
    }
    return df[channel_dict[channel]].iloc[-1]

def technical_channel_account_compare(df: pd.DataFrame, ohlc: str, compare: str, channel: str, **kwargs):
    channel_dict = {
        'upper': 'Upper',
        'middle': 'Middle',
        'lower': 'Lower',
    }
    channel = channel_dict[channel]
    count = 0
    for i in range(len(df)):
        if compare == 'greater' and df[ohlc].iloc[i] > df[channel].iloc[i]:
            count += 1
        elif compare == 'less' and df[ohlc].iloc[i] < df[channel].iloc[i]:
            count += 1
    return count / len(df) * 100

question_template_technical_channel = [
    {
        'templates': [
            'What is the {channel} channel value of {indicator} for {corporate} on {date}?',
            'On {date}, what is the {indicator} {channel} channel value for {corporate}?',
            'For {corporate}\'s {indicator} on {date}, what is the {channel} value?',
            'What is the value of the {channel} of {indicator} for {corporate} on {date}?',
            'Calculate the {indicator} {channel} channel value for {corporate} on {date}.',
        ],
        'difficulty': 'easy',
        'function': technical_channel_value,  # single-point calculation
    },
    {
        'templates': [
            'What is the percentage of days where the {ohlc} price is {compare} than the {channel} channel of {indicator} for {corporate} from {start_date} to {end_date}? (Unit: %)',
            'From {start_date} to {end_date}, what proportion of days had {corporate}\'s {ohlc} {compare} than the {channel} of {indicator}? (Unit: %)',
            'During {start_date} to {end_date}, what percentage of days had {corporate}\'s {ohlc} price {compare} the {channel} channel of {indicator}?',
            'For {corporate} from {start_date} to {end_date}, what percentage of days had the {ohlc} {compare} than the {indicator} {channel} channel? (Unit: %)',
            'Calculate the percentage of days where {corporate}\'s {ohlc} was {compare} than the {channel} channel of {indicator} from {start_date} to {end_date}.',
        ],
        'difficulty': 'medium',
        'function': technical_channel_account_compare,  # pattern: above/below
    },
    # 'What is the {n}-day period {indicator} {channel} value for {corporate} on {date}?',  # single-point calculation
    # 'What is the percentage of days where the {ohlc} price is {compare} than the {channel} of the {n}-day period {indicator} for {corporate} from {start_date} to {end_date}?',  # pattern: above/below
]

CHANNEL_LIST = ['upper', 'middle', 'lower']


# question_list_technical_n = [
#     'What is the value of {indicator} for {corporate} on {date}?',  # single-point calculation
# ]

question_template_technical_other = [
    {
        'templates': [
            'What is the value of {indicator} for {corporate} on {date}?',
            'On {date}, what is the {indicator} value for {corporate}?',
            'What was {corporate}\'s {indicator} on {date}?',
            'For {indicator}, what is the value of {corporate} on {date}?',
            'Calculate the {indicator} value for {corporate} on {date}.',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # single-point calculation
    },
]


question_template_technical_performance = [
    {
        'templates': [
            'What is the value of {indicator} for {corporate} from {start_date} to {end_date}?',
            'From {start_date} to {end_date}, what is {corporate}\'s {indicator}?',
            'What was {corporate}\'s {indicator} during the period {start_date} to {end_date}?',
            'For {indicator}, what is {corporate}\'s value from {start_date} to {end_date}?',
            'Calculate {corporate}\'s {indicator} from {start_date} to {end_date}.',
        ],
        'difficulty': 'easy',
        'function': lambda x, **kwargs: x,  # single-point calculation
    },
]


def calculate_max_mean_diff(df, indicator1: str, **kwargs):
    return df[indicator1].max() - df[indicator1].mean()


question_template_macro = [
    {
        'templates': [
            'What is the {indicator} of the US in {year}?',
            'In {year}, what was the US {indicator}?',
            'What was the {indicator} reported in the US in {year}?',
            'For {indicator}, what is the US value in {year}?',
            'Calculate the US {indicator} for {year}.',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # single-point calculation
    },
    {
        'templates': [
            'What is the {statistic} of {indicator} in the US from {start_year} to {end_year}?',
            'From {start_year} to {end_year}, what is the {statistic} of the US {indicator}?',
            'What was the {statistic} of {indicator} in the US during {start_year}–{end_year}?',
            'For US {indicator}, what is the {statistic} over {start_year} to {end_year}?',
            'Calculate the {statistic} of the US {indicator} from {start_year} to {end_year}.',
        ],
        'difficulty': 'medium',
        'function': calculate_statistic,  # period statistics
    },
    {
        'templates': [
            'What is the difference between the maximum and average values of the US {indicator1} from {start_year} to {end_year}?',
            'From {start_year} to {end_year}, what was the difference between the maximum and average values of the US {indicator1}?',
            'During {start_year}–{end_year}, how much did the US {indicator1}\'s maximum value exceed its average?',
            'In the period {start_year}–{end_year}, what was the gap between the highest and average values of the US {indicator1}?',
            'For the US {indicator1} from {start_year} to {end_year}, calculate the difference between its maximum and average values.',
        ],
        'difficulty': 'hard',
        'function': calculate_max_mean_diff,  # 差值计算
    },
]

question_suffix_macro = ' Write down the indicator formula, collect the raw data, calculate the answer, and keep four decimal places. Please list the raw data and key calculation process.'


STATISTIC_MACRO_LIST = ['average', 'maximum', 'minimum']
