import datetime
import pandas as pd
from QACalculator import IndicatorError



def calculate_statistic(df: pd.DataFrame, indicator: str, statistic: str, **kwargs):
    df_indicator_col = df[indicator]
    if statistic == '平均值':
        return df_indicator_col.mean()
    elif statistic == '最大值':
        return df_indicator_col.max()
    elif statistic == '最小值':
        return df_indicator_col.min()
    elif statistic == '复合年均增长率':
        start_value = df_indicator_col.iloc[0]
        end_value = df_indicator_col.iloc[-1]
        n = len(df_indicator_col) - 1
        if start_value <= 0 or end_value <= 0:
            raise IndicatorError("CAGR is not defined for non-positive values.")
        return ((end_value / start_value) ** (1/n) - 1) * 100
    elif statistic in ['同比增长率', '环比增长率']:
        start_value = df_indicator_col.iloc[0]
        end_value = df_indicator_col.iloc[-1]
        if start_value <= 0:
            raise IndicatorError("Growth rate is not defined for non-positive values.")
        return (end_value - start_value) / start_value * 100
    else:
        raise IndicatorError(f"Unsupported statistic: {statistic}")

def calculate_statistic_and_compare(df: pd.DataFrame, statistic: str, indicator1: str, corporate1: str, corporate2: str, **kwargs):
    # print('---\n', df)
    stat1 = calculate_statistic(df.query(f'corporate == "{corporate1}"'), indicator1, statistic)
    stat2 = calculate_statistic(df.query(f'corporate == "{corporate2}"'), indicator1, statistic)
    if stat2 == 0:
        raise IndicatorError('Cannot compare when the second statistic is zero.')
    return stat1 / stat2

def indicator_and_compare(df: pd.DataFrame, indicator: str, high_or_low: str, **kwargs):
    if high_or_low == '高':
        best_corp = df.idxmax()
        best_value = df.max()
        second_value = df[indicator].nlargest(2).iloc[-1]
        return best_value - second_value
    elif high_or_low == '低':
        best_corp = df.idxmin()
        best_value = df.min()
        second_value = df[indicator].nsmallest(2).iloc[-1]
        return second_value - best_value
    else:
        raise IndicatorError(f"Unsupported high_or_low: {high_or_low}")

def indicator_another(df: pd.DataFrame, indicator1: str, high_or_low: str, indicator2: str, **kwargs):
    if high_or_low == '高':
        best_year = df[indicator1].idxmax()
    elif high_or_low == '低':
        best_year = df[indicator1].idxmin()
    else:
        raise IndicatorError(f"Unsupported high_or_low: {high_or_low}")
    return df.loc[best_year, indicator2]

def count_change(df: pd.DataFrame, indicator1: str, change: str, **kwargs):
    count = 0
    for i in range(1, len(df)):
        if change == '上升' and df[indicator1].iloc[i] > df[indicator1].iloc[i-1]:
            count += 1
        elif change == '下降' and df[indicator1].iloc[i] < df[indicator1].iloc[i-1]:
            count += 1
    return count


question_template_fundamental = [
    # '{corporate}{year}{quarter}的{indicator}是？',  # 单点计算
    # '{corporate}{year}{quarter}的{indicator1}的{statistic_quarter}是？（单位：%）',  # 时点统计值
    {
        'templates': [
            '{corporate}{year}年的{indicator}是？',
            '{year}年{corporate}的{indicator}是？',
            '{corporate}的{indicator}在{year}年是多少？',
            '对于{indicator}，{corporate}在{year}年的值是？',
            '计算{year}年度{corporate}的{indicator}。',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # 单点计算
    },
    {
        'templates': [
            '{corporate}{start_year}-{end_year}年的{indicator}的{statistic}是？',
            '{start_year}至{end_year}年间，{corporate}{indicator}的{statistic}是多少？',
            '{corporate}的{indicator}在{start_year}到{end_year}年的{statistic}是？',
            '对于{corporate}的{indicator}，其{start_year}-{end_year}年的{statistic}是多少？',
            '计算{corporate}在{start_year}-{end_year}年的{indicator}的{statistic}。',
        ],
        'difficulty': 'medium',
        'function': calculate_statistic,  # 时段统计值
    },
    {
        'templates': [
            '{start_year}-{end_year}年的{indicator1}的{statistic}, {corporate1}与{corporate2}之比是多少？',
            '{corporate1}与{corporate2}在{start_year}至{end_year}年{indicator1}{statistic}之比是多少？',
            '{start_year}到{end_year}年，{corporate1}的{indicator1}{statistic}是{corporate2}的几倍？',
            '在{start_year}-{end_year}年间，{indicator1}的{statistic}，{corporate1}相对{corporate2}的比值是多少？',
            '分别统计{corporate1}和{corporate2}在{start_year}-{end_year}年的{indicator1}的{statistic}，计算两者之比。',
        ],
        'difficulty': 'hard',
        'function': calculate_statistic_and_compare,  # 跨公司比较
    },
    {
        'templates': [
            '在{corporates}中，{year}年{indicator}最{high_or_low}的那家公司，比第二名{high_or_low}出多少？',
            '{corporates}里，{year}年{indicator}最{high_or_low}的企业与第二名相差多少？',
            '{year}年，在{corporates}中，{indicator}最{high_or_low}的公司与第二名之间的差距是多少？',
            '对于{year}年的{indicator}，{corporates}中最{high_or_low}的那家公司领先第二名多少？',
            '对于{corporates}在{year}年{indicator}的值，计算最{high_or_low}的公司与第二名之间的差距。',
        ],
        'difficulty': 'hard',
        'function': indicator_and_compare,  # 跨公司比较
    },
    # '{start_year}-{end_year}年, {corporate}的{indicator1}最{high_or_low}的那一年，{indicator2}是多少？': indicator_another,  # 极值定位 + 衍生计算
    {
        'templates': [
            '计算{corporate}在{start_year}至{end_year}年间的{indicator1}，做同比比较，其中同比{change}有几次？',
            '{start_year}至{end_year}年，{corporate}的{indicator1}逐年同比，其中同比{change}共多少次？',
            '{corporate}在{start_year}到{end_year}年间，逐年做{indicator1}同比，同比{change}的年份有多少个？',
            '对{corporate}{start_year}-{end_year}年的{indicator1}做逐年同比，共发生同比{change}几次？',
            '统计{corporate}的{indicator1}在{start_year}-{end_year}年间逐年同比{change}的次数。',
        ],
        'difficulty': 'hard',
        'function': count_change,  # 变化趋势统计
    }
]

question_suffix_fundamental = '按中国会计准则，写出指标公式，搜集原始数据，计算得出答案，保留四位小数。注意列出原始数据和关键计算过程。'

df = pd.read_csv('data/hs300_list_2025-12.csv')
CORPORATE_LIST = df['code_name'].tolist()
# corporate_list = ['宁德时代', '中际旭创', '贵州茅台', '中国平安', '紫金矿业', '美的集团', '长江电力', '中国神华', '寒武纪', '恒瑞医药']
# corporate_list = ['宁德时代', '贵州茅台']

QUARTER_LIST = ['Q1', 'Q2', 'Q3', 'Q4']

END_YEAR_LIST = list(range(2016, 2026))
T_YEAR_LIST = [2, 4]

STATISTIC_QUARTER_LIST = ['同比增长率', '环比增长率']
STATISTIC_LIST = ['平均值', '最大值', '最小值', '复合年均增长率']
UNIT1_LIST = ['%']
UNIT2_LIST = ['亿元', '元']

HIGH_OR_LOW_LIST = ['高', '低']
CHANGE_LIST = ['上升', '下降']



def technical_ma_count_x(df: pd.DataFrame, n1: int, n2: int, x: str, **kwargs):
    n_fast, n_slow = min(n1, n2), max(n1, n2)
    count = 0
    if x == '金叉':
        for i in range(1, len(df)):
            if df[f'{n_fast}日'].iloc[i] > df[f'{n_slow}日'].iloc[i] and df[f'{n_fast}日'].iloc[i-1] <= df[f'{n_slow}日'].iloc[i-1]:
                count += 1
    elif x == '死叉':
        for i in range(1, len(df)):
            if df[f'{n_fast}日'].iloc[i] < df[f'{n_slow}日'].iloc[i] and df[f'{n_fast}日'].iloc[i-1] >= df[f'{n_slow}日'].iloc[i-1]:
                count += 1
    return count

def technical_ma_account_compare(df: pd.DataFrame, ohlc: str, compare: str, n: int, **kwargs):
    ohlc_dict = {
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
    }
    ohlc = ohlc_dict[ohlc]
    count = 0
    for i in range(len(df)):
        if compare == '大于' and df[f'{n}日'].iloc[i] > df[ohlc].iloc[i]:
            count += 1
        elif compare == '小于' and df[f'{n}日'].iloc[i] < df[ohlc].iloc[i]:
            count += 1
    return count / len(df) * 100

def technical_ma_count_breakthrough(df: pd.DataFrame, breakthrough: str, n: int, **kwargs):
    count = 0
    for i in range(1, len(df)):
        if breakthrough == '突破' and df['close'].iloc[i] > df[f'{n}日'].iloc[i] and df['close'].iloc[i-1] < df[f'{n}日'].iloc[i-1]:
            count += 1
        elif breakthrough == '跌破' and df['close'].iloc[i] < df[f'{n}日'].iloc[i] and df['close'].iloc[i-1] > df[f'{n}日'].iloc[i-1]:
            count += 1
    return count

def technical_ma_account_long(df: pd.DataFrame, ns: list, long: str, **kwargs):
    n_list = sorted(list(map(int, ns.split('、'))))
    count = 0
    for i in range(len(df)):
        if long == '多头' and all(df[f'{n}日'].iloc[i] > df[f'{m}日'].iloc[i] for n, m in zip(n_list[:-1], n_list[1:])):
            count += 1
        elif long == '空头' and all(df[f'{n}日'].iloc[i] < df[f'{m}日'].iloc[i] for n, m in zip(n_list[:-1], n_list[1:])):
            count += 1
    return count / len(df) * 100

question_template_technical_ma = [
    {
        'templates': [
            '{corporate}{date}的{n}日周期{indicator}收盘价是？（单位：元）',
            '{date}，{corporate}的{n}日{indicator}收盘价为多少元？',
            '对于{n}日周期{indicator}收盘价，{date}{corporate}的值是多少？（单位：元）',
            '对于{corporate}的{n}日{indicator}收盘价（单位：元），{date}的值是多少？',
            '计算{corporate}{date}的{n}日周期{indicator}收盘价。（单位：元）',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # 单点计算
    },
    # '{corporate}{date}的{n}日周期{indicator}对数收益率是？',  # 单点计算
    {
        'templates': [
            '{corporate}{start_date}至{end_date}, 对于{indicator}的{n1}日线和{n2}日线，有多少次{x}？',
            '{corporate}从{start_date}到{end_date}，{indicator}的{n1}日线与{n2}日线发生{x}多少次？',
            '在{start_date}至{end_date}期间，{corporate}的{indicator}{n1}日线和{n2}日线共出现几次{x}？',
            '{start_date}-{end_date}，{corporate}{indicator}线中{n1}日均线和{n2}日均线{x}的次数是多少？',
            '对于{corporate}在{start_date}到{end_date}间{indicator}线，统计{n1}日线与{n2}日线的{x}次数。',
        ],
        'difficulty': 'medium',
        'function': technical_ma_count_x,  # 形态：金叉/死叉
    },
    {
        'templates': [
            '{corporate}{start_date}至{end_date}, {ohlc}{compare}{n}日{indicator}线的天数占比？（单位：%）',
            '{corporate}从{start_date}到{end_date}，{ohlc}{compare}{n}日{indicator}线的天数比例是多少？（单位：%）',
            '在{start_date}至{end_date}期间，对于{corporate}的{indicator}线，{ohlc}{compare}{n}日线的天数占总天数的百分比？',
            '{start_date}-{end_date}，{corporate}的{ohlc}{compare}{n}日{indicator}线的天数占比多少？（%）',
            '统计{corporate}在{start_date}到{end_date}间，{ohlc}{compare}{n}日{indicator}线的天数占比（%）。',
        ],
        'difficulty': 'medium',
        'function': technical_ma_account_compare,  # 形态：多空占比
    },
    {
        'templates': [
            '{corporate}{start_date}至{end_date}, 价格{breakthrough}{n}日{indicator}线有多少次？',
            '{corporate}在{start_date}到{end_date}期间，收盘价{breakthrough}{n}日{indicator}线共几次？',
            '{start_date}至{end_date}，{corporate}股价{breakthrough}{n}日{indicator}线的次数是多少？',
            '在{start_date}-{end_date}，{corporate}的价格有多少次{breakthrough}{n}日{indicator}线？',
            '统计{corporate}从{start_date}到{end_date}，股价{breakthrough}{n}日{indicator}线的总次数。',
        ],
        'difficulty': 'medium',
        'function': technical_ma_count_breakthrough,  # 形态：突破/跌破
    },
    {
        'templates': [
            '{corporate}{start_date}至{end_date}, 对于{ns}日{indicator}线，{long}排列的天数占比？（单位：%）',
            '{corporate}从{start_date}到{end_date}，{ns}日{indicator}线呈{long}排列的天数比例是多少？（单位：%）',
            '{start_date}-{end_date}，{corporate}的{ns}日{indicator}线处于{long}排列的天数占比多少？（单位：%）',
            '在{start_date}至{end_date}，{corporate}{ns}日{indicator}线{long}排列的天数占总天数的百分比？',
            '统计{corporate}在{start_date}到{end_date}的{ns}日{indicator}线{long}排列的天数占比（%）。',
        ],
        'difficulty': 'hard',
        'function': technical_ma_account_long,  # 形态：多头/空头排列
    },
]

question_suffix_technical = '使用不复权价格，写出指标公式，搜集原始数据，计算得出答案，保留四位小数。注意列出原始数据和关键计算过程。'

END_DATE_LIST = pd.date_range(start='2020-01-01', end='2025-12-31').tolist()
T_DAY_LIST = list(range(10, 366))
# start_date = end_date - datetime.timedelta(days=t_day)

X_LIST = ['金叉', '死叉']
N_LIST = [5, 10, 20, 30, 60]
OHLC_LIST = ['开盘价', '最高价', '最低价', '收盘价']
COMPARE_LIST = ['大于', '小于']
BREAKTHROUGH_LIST = ['突破', '跌破']
NS_LIST = ['5、10、20', '10、20、30', '5、20、60']
LONG_LIST = ['多头', '空头']


def technical_channel_value(df: pd.DataFrame, channel: str, **kwargs):
    channel_dict = {
        '上轨': 'Upper',
        '中轨': 'Middle',
        '下轨': 'Lower',
    }
    return df[channel_dict[channel]].iloc[-1]

def technical_channel_account_compare(df: pd.DataFrame, ohlc: str, compare: str, channel: str, **kwargs):
    channel_dict = {
        '上轨': 'Upper',
        '中轨': 'Middle',
        '下轨': 'Lower',
    }
    channel = channel_dict[channel]
    ohlc_dict = {
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
    }
    ohlc = ohlc_dict[ohlc]
    count = 0
    for i in range(len(df)):
        if compare == '大于' and df[ohlc].iloc[i] > df[channel].iloc[i]:
            count += 1
        elif compare == '小于' and df[ohlc].iloc[i] < df[channel].iloc[i]:
            count += 1
    return count / len(df) * 100

question_template_technical_channel = [
    {
        'templates': [
            '{corporate}{date}的{indicator}{channel}的值是？',
            '{date}，{corporate}的{indicator}{channel}值是多少？',
            '对于{corporate}{date}的{indicator}，其{channel}值是多少？',
            '关于{indicator}的{channel}，{date}{corporate}的值是？',
            '计算{corporate}{date}的{indicator}{channel}值。',
        ],
        'difficulty': 'easy',
        'function': technical_channel_value,  # 单点计算
    },
    {
        'templates': [
            '{corporate}{start_date}-{end_date}, 对于{indicator}，{ohlc}{compare}{channel}的天数占比？（单位：%）',
            '{corporate}从{start_date}到{end_date}，{ohlc}{compare}{indicator}{channel}的天数比例是多少？（单位：%）',
            '在{start_date}至{end_date}期间，{corporate}{ohlc}{compare}{indicator}{channel}的天数占总天数的百分比？',
            '{start_date}-{end_date}，{corporate}的{ohlc}{compare}{indicator}{channel}的比例（单位：%）？',
            '统计{corporate}在{start_date}到{end_date}间，{ohlc}{compare}{indicator}{channel}的天数占比（%）。',
        ],
        'difficulty': 'medium',
        'function': technical_channel_account_compare,  # 形态：上下方
    },
    # '{corporate}{date}的{n}日周期{indicator}{channel}的值是？',  # 单点计算
    # '{corporate}{start_date}-{end_date}, 对于{n}日周期{indicator}，{ohlc}{compare}{channel}的天数占比？',  # 形态：上下方
]

CHANNEL_LIST = ['上轨', '中轨', '下轨']


# question_list_technical_n = [
#     '{corporate}{date}的{n}日周期{indicator}的值是？',  # 单点计算
# ]

question_template_technical_other = [
    {
        'templates': [
            '{corporate}{date}的{indicator}的值是？',
            '{date}，{corporate}的{indicator}是多少？',
            '{corporate}的{indicator}在{date}为多少？',
            '对于{indicator}，{corporate}{date}的值是多少？',
            '计算{corporate}{date}的{indicator}的值。',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # 单点计算
    },
]


question_template_technical_performance = [
    {
        'templates': [
            '{corporate}{start_date}至{end_date}的{indicator}的值是？',
            '{start_date}到{end_date}，{corporate}的{indicator}是多少？',
            '{corporate}的{indicator}，在{start_date}-{end_date}期间为多少？',
            '对于{start_date}-{end_date}的{indicator}，{corporate}的值是？',
            '计算{corporate}{start_date}至{end_date}的{indicator}。',
        ],
        'difficulty': 'easy',
        'function': lambda x, **kwargs: x,  # 单点计算
    },
]


def calculate_max_mean_diff(df, indicator1: str, **kwargs):
    return df[indicator1].max() - df[indicator1].mean()

question_template_macro = [
    {
        'templates': [
            '中国{year}年的{indicator}是？',
            '{year}年中国的{indicator}是？',
            '中国的{indicator}在{year}年是多少？',
            '对于{indicator}，中国在{year}年的值是？',
            '计算{year}年度中国的{indicator}。',
        ],
        'difficulty': 'easy',
        'function': lambda df, **kwargs: df.iloc[-1, -1],  # 单点计算
    },
    {
        'templates': [
            '中国{start_year}-{end_year}年的{indicator}的{statistic}是？',
            '{start_year}至{end_year}年间，中国{indicator}的{statistic}是多少？',
            '中国的{indicator}在{start_year}到{end_year}年的{statistic}是？',
            '对于中国的{indicator}，其{start_year}-{end_year}年的{statistic}是多少？',
            '计算中国在{start_year}至{end_year}年的{indicator}的{statistic}。',
        ],
        'difficulty': 'medium',
        'function': calculate_statistic,  # 时段统计值
    },
    {
        'templates': [
            '中国在{start_year}至{end_year}年间的{indicator1}，最大值比平均值高出多少？',
            '{start_year}至{end_year}年，中国的{indicator1}的最大值与平均值之差是多少？',
            '中国在{start_year}到{end_year}年间，{indicator1}的最大值与平均值相差多少？',
            '在{start_year}-{end_year}年间，对于中国的{indicator1}，最大值领先平均值多少？',
            '对于中国在{start_year}-{end_year}年的{indicator1}，计算最大值与平均值的差。',
        ],
        'difficulty': 'hard',
        'function': calculate_max_mean_diff,  # 差值计算
    },
]

question_suffix_macro = '写出指标公式，搜集原始数据，计算得出答案，保留四位小数。注意列出原始数据和关键计算过程。'


STATISTIC_MACRO_LIST = ['平均值', '最大值', '最小值']