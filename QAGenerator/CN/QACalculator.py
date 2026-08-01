import numpy as np
import pandas as pd
from scipy.stats import norm



class IndicatorError(Exception):
    pass

def check_condition(condition):
    if not condition.all():
        raise IndicatorError('Data quality check failed!')
    return True

indicators_fundamental = {
    # 盈利能力
    '毛利率': {
        'sub-category': 'profitability',
        'data': ['营业收入', '营业成本'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            毛利率=lambda df: (df['营业收入'] - df['营业成本']) / df['营业收入'] * 100,
        )['毛利率'],
        'formula': '(营业收入 - 营业成本) / 营业收入 * 100%',
        'note': '单位：%'
    },
    'EBIT利润率': {
        'sub-category': 'profitability',
        'data': ['营业收入', '利润总额', '财务费用——利息费用'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            EBIT利润率=lambda df: (df['利润总额'] + df['财务费用——利息费用']) / df['营业收入'] * 100
        )['EBIT利润率'],
        'formula': '(利润总额 + 利息支出) / 营业收入 * 100%',
        'note': '单位：%'
    },
    'EBITDA利润率': {
        'sub-category': 'profitability',
        'data': ['营业收入', '利润总额', '财务费用——利息费用', '固定资产折旧、油气资产折耗、生产性生物资产折旧', '无形资产摊销', '长期待摊费用摊销'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            EBITDA利润率=lambda df: (df['利润总额'] + df['财务费用——利息费用'] + df['固定资产折旧、油气资产折耗、生产性生物资产折旧'] + df['无形资产摊销'] + df['长期待摊费用摊销']) / df['营业收入'] * 100
        )['EBITDA利润率'],
        'formula': '(利润总额 + 利息支出 + 折旧与摊销) / 营业收入 * 100%',
        'note': '单位：%'
    },
    '净利率': {
        'sub-category': 'profitability',
        'data': ['营业收入', '净利润'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            净利率=lambda df: df['净利润'] / df['营业收入'] * 100
        )['净利率'],
        'formula': '净利润 / 营业收入 * 100%',
        'note': '单位：%'
    },
    '净资产收益率': {
        'sub-category': 'profitability',
        'data': ['归属于母公司所有者的净利润', '归属于母公司所有者权益合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['归属于母公司所有者权益合计'] > 0),
            净资产收益率=lambda df: df['归属于母公司所有者的净利润'] / df['归属于母公司所有者权益合计'] * 100
        )['净资产收益率'],
        'note': '单位：%；按归属于母公司的计算；所有者权益按期末值计算',
        'formula': '净利润 / 所有者权益 * 100%'
    },
    '总资产收益率': {
        'sub-category': 'profitability',
        'data': ['净利润', '资产合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['资产合计'] > 0),
            总资产收益率=lambda df: df['净利润'] / df['资产合计'] * 100
        )['总资产收益率'],
        'note': '单位：%；总资产按期末值计算',
        'formula': '净利润 / 总资产 * 100%',
    },
    '投入资本回报率': {
        'sub-category': 'profitability',
        'data': ['利润总额', '财务费用——利息费用', '所得税费用', '股东权益合计', '短期借款', '长期借款', '应付债券', '一年内到期的非流动负债', '租赁负债', '应付利息', '货币资金', '交易性金融资产', '其他权益工具投资', '其他债权投资', '债权投资', '衍生金融资产', '买入返售金融资产'],
        'function':  lambda df: df.assign(
            CHECK1=lambda df: check_condition((df['所得税费用'] >= 0) & (df['利润总额'] > 0)),
            NOPAT=lambda df: (df['利润总额'] + df['财务费用——利息费用']) * (1 - df['所得税费用'] / df['利润总额']),
            投入资本=lambda df: df['股东权益合计'] + df['短期借款'] + df['长期借款'] + df['应付债券'] + df['一年内到期的非流动负债'] + df['租赁负债'] + df['应付利息'] - df['货币资金'] - df['交易性金融资产'] - df['其他权益工具投资'] - df['其他债权投资'] - df['债权投资'] - df['衍生金融资产'] - df['买入返售金融资产'],
            CHECK2=lambda df: check_condition(df['投入资本'] > 0),
            ROIC=lambda df: df['NOPAT'] / ((df['投入资本'].shift(1) + df['投入资本']) / 2) * 100
        )['ROIC'],
        'note': '单位：%；投入资本用期初期末平均值',
        'formula': 'ROIC = NOPAT / 平均投入资本 * 100%；NOPAT = (利润总额 + 利息支出) × (1 - 所得税费用 / 利润总额)；投入资本 = 股东权益 + 有息负债（短期借款 + 长期借款 + 应付债券 + 一年内到期非流动负债 + 租赁负债 + 应付利息）- 金融资产（货币资金 + 交易性金融资产 + 其他权益工具投资 + 其他债权投资 + 债权投资 + 衍生金融资产 + 买入返售金融资产）'
    },
    # 营运能力
    '存货周转天数': {
        'sub-category': 'operational efficiency',
        'data': ['营业成本', '存货'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业成本'] > 0),
            存货周转天数=lambda df: (df['存货'].shift(1) + df['存货']) / 2 / df['营业成本'] * 365
        )['存货周转天数'],
        'note': '存货用期初期末平均值；一年按365天计算',
        'formula': '平均存货 / 营业成本 * 365'
    },
    '应收账款周转天数': {
        'sub-category': 'operational efficiency',
        'data': ['营业收入', '应收票据及应收账款'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            应收账款周转天数=lambda df: (df['应收票据及应收账款'].shift(1) + df['应收票据及应收账款']) / 2 / df['营业收入'] * 365
        )['应收账款周转天数'],
        'note': '使用应收票据及应收账款，用期初期末平均值；一年按365天计算',
        'formula': '平均应收票据及应收账款 / 营业收入 * 365'
    },
    '应付账款周转天数': {
        'sub-category': 'operational efficiency',
        'data': ['营业成本', '应付票据及应付账款'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业成本'] > 0),
            应付账款周转天数=lambda df: (df['应付票据及应付账款'].shift(1) + df['应付票据及应付账款']) / 2 / df['营业成本'] * 365
        )['应付账款周转天数'],
        'note': '使用应付票据及应付账款，用期初期末平均值；一年按365天计算',
        'formula': '平均应付票据及应付账款 / 营业成本 * 365'
    },
    '营业周期': {
        'sub-category': 'operational efficiency',
        'data': ['营业成本', '存货', '营业收入', '应收票据及应收账款'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['营业成本'] > 0) & (df['营业收入'] > 0)),
            营业周期=lambda df: (df['存货'].shift(1) + df['存货']) / 2 / df['营业成本'] * 365 + (df['应收票据及应收账款'].shift(1) + df['应收票据及应收账款']) / 2 / df['营业收入'] * 365
        )['营业周期'],
        'note': '计算应收账款周转天数时，使用应收票据及应收账款；存货和应收票据及应收账款都用期初期末平均值；一年按365天计算',
        'formula': '存货周转天数 + 应收账款周转天数'
    },
    '现金周转期': {
        'sub-category': 'operational efficiency',
        'data': ['营业成本', '存货', '营业收入', '应收票据及应收账款', '应付票据及应付账款'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['营业成本'] > 0) & (df['营业收入'] > 0)),
            现金周转期=lambda df: (df['存货'].shift(1) + df['存货']) / 2 / df['营业成本'] * 365 + (df['应收票据及应收账款'].shift(1) + df['应收票据及应收账款']) / 2 / df['营业收入'] * 365 - (df['应付票据及应付账款'].shift(1) + df['应付票据及应付账款']) / 2 / df['营业成本'] * 365
        )['现金周转期'],
        'note': '计算应收账款周转天数时，使用应收票据及应收账款；计算应付账款周转天数时，使用应付票据及应付账款；存货、应收票据及应收账款和应付票据及应付账款都用期初期末平均值；一年按365天计算',
        'formula': '存货周转天数 + 应收账款周转天数 - 应付账款周转天数'
    },
    '存货周转率': {
        'sub-category': 'operational efficiency',
        'data': ['营业成本', '存货'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['存货'] > 0),
            存货周转率=lambda df: df['营业成本'] / ((df['存货'].shift(1) + df['存货']) / 2)
        )['存货周转率'],
        'note': '存货用期初期末平均值',
        'formula': '营业成本 / 平均存货'
    },
    '应收账款周转率': {
        'sub-category': 'operational efficiency',
        'data': ['营业收入', '应收票据及应收账款'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['应收票据及应收账款'] > 0),
            应收账款周转率=lambda df: df['营业收入'] / ((df['应收票据及应收账款'].shift(1) + df['应收票据及应收账款']) / 2)
        )['应收账款周转率'],
        'note': '使用应收票据及应收账款，用期初期末平均值',
        'formula': '营业收入 / 平均应收票据及应收账款'
    },
    '应付账款周转率': {
        'sub-category': 'operational efficiency',
        'data': ['营业成本', '应付票据及应付账款'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['应付票据及应付账款'] > 0),
            应付账款周转率=lambda df: df['营业成本'] / ((df['应付票据及应付账款'].shift(1) + df['应付票据及应付账款']) / 2)
        )['应付账款周转率'],
        'note': '使用应付票据及应付账款，用期初期末平均值',
        'formula': '营业成本 / 平均应付票据及应付账款'
    },
    '总资产周转率': {
        'sub-category': 'operational efficiency',
        'data': ['营业收入', '资产合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['资产合计'] > 0),
            总资产周转率=lambda df: df['营业收入'] / ((df['资产合计'].shift(1) + df['资产合计']) / 2)
        )['总资产周转率'],
        'note': '总资产按期初期末平均值计算',
        'formula': '营业收入 / 平均总资产'
    },
    '固定资产周转率': {
        'sub-category': 'operational efficiency',
        'data': ['营业收入', '固定资产合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['固定资产合计'] > 0),
            固定资产周转率=lambda df: df['营业收入'] / ((df['固定资产合计'].shift(1) + df['固定资产合计']) / 2)
        )['固定资产周转率'],
        'note': '固定资产用期初期末平均值',
        'formula': '营业收入 / 平均固定资产合计'
    },
    '营运资本周转率': {
        'sub-category': 'operational efficiency',
        'data': ['营业收入', '流动资产合计', '流动负债合计'],
        'function': lambda df: df.assign(
            净营运资本=lambda df: df['流动资产合计'] - df['流动负债合计'],
            平均净营运资本=lambda df: (df['净营运资本'].shift(1) + df['净营运资本']) / 2,
            CHECK=lambda df: check_condition(df['平均净营运资本'] != 0),
            营运资本周转率=lambda df: df['营业收入'] / df['平均净营运资本']
        )['营运资本周转率'],
        'note': '净营运资本用期初期末平均值',
        'formula': '营业收入 / 平均净营运资本；净营运资本 = 流动资产合计 - 流动负债合计'
    },
    # 偿债能力
    '流动比率': {
        'sub-category': 'solvency',
        'data': ['流动资产合计', '流动负债合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['流动负债合计'] > 0),
            流动比率=lambda df: df['流动资产合计'] / df['流动负债合计']
        )['流动比率'],
        'formula': '流动资产合计 / 流动负债合计'
    },
    '速动比率': {
        'sub-category': 'solvency',
        'data': ['流动资产合计', '流动负债合计', '存货'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['流动负债合计'] > 0),
            速动比率=lambda df: (df['流动资产合计'] - df['存货']) / df['流动负债合计']
        )['速动比率'],
        'formula': '(流动资产合计 - 存货) / 流动负债合计'
    },
    '资产负债率': {
        'sub-category': 'solvency',
        'data': ['负债合计', '资产合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['资产合计'] > 0),
            资产负债率=lambda df: df['负债合计'] / df['资产合计'] * 100
        )['资产负债率'],
        'note': '单位：%',
        'formula': '负债合计 / 资产合计 * 100%'
    },
    '产权比率': {
        'sub-category': 'solvency',
        'data': ['负债合计', '股东权益合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['股东权益合计'] > 0),
            产权比率=lambda df: df['负债合计'] / df['股东权益合计']
        )['产权比率'],
        'formula': '负债合计 / 股东权益合计'
    },
    '净债务权益比': {
        'sub-category': 'solvency',
        'data': ['短期借款', '一年内到期的非流动负债', '长期借款', '应付债券', '租赁负债', '应付利息', '应付短期债券', '交易性金融负债', '卖出回购金融资产款', '货币资金', '交易性金融资产', '股东权益合计'],
        'function': lambda df: df.assign(
            净债务=lambda df: df['短期借款'] + df['一年内到期的非流动负债'] + df['长期借款'] + df['应付债券'] + df['租赁负债'] + df['应付利息'] + df['应付短期债券'] + df['交易性金融负债'] + df['卖出回购金融资产款'] - df['货币资金'] - df['交易性金融资产'],
            CHECK=lambda df: check_condition(df['股东权益合计'] > 0),
            净债务权益比=lambda df: df['净债务'] / df['股东权益合计']
        )['净债务权益比'],
        'formula': '净债务权益比 = 净债务 / 股东权益合计；净债务 = 有息负债（短期借款 + 一年内到期非流动负债 + 长期借款 + 应付债券 + 租赁负债 + 应付利息 + 应付短期债券 + 交易性金融负债 + 卖出回购金融资产款）- 现金类资产（货币资金 + 交易性金融资产）'
    },
    '利息保障倍数': {
        'sub-category': 'solvency',
        'data': ['利润总额', '财务费用——利息费用'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['财务费用——利息费用'] > 0),
            利息保障倍数=lambda df: (df['利润总额'] + df['财务费用——利息费用']) / df['财务费用——利息费用']
        )['利息保障倍数'],
        'formula': '利息保障倍数 = EBIT / 利息支出 = (利润总额 + 利息支出) / 利息支出'
    },
    '净债务/EBITDA': {
        'sub-category': 'solvency',
        'data': ['短期借款', '一年内到期的非流动负债', '长期借款', '应付债券', '租赁负债', '应付利息', '应付短期债券', '交易性金融负债', '卖出回购金融资产款', '货币资金', '交易性金融资产', '利润总额', '财务费用——利息费用', '固定资产折旧、油气资产折耗、生产性生物资产折旧', '无形资产摊销', '长期待摊费用摊销'],
        'function': lambda df: df.assign(
            净债务=lambda df: df['短期借款'] + df['一年内到期的非流动负债'] + df['长期借款'] + df['应付债券'] + df['租赁负债'] + df['应付利息'] + df['应付短期债券'] + df['交易性金融负债'] + df['卖出回购金融资产款'] - df['货币资金'] - df['交易性金融资产'],
            EBITDA=lambda df: df['利润总额'] + df['财务费用——利息费用'] + df['固定资产折旧、油气资产折耗、生产性生物资产折旧'] + df['无形资产摊销'] + df['长期待摊费用摊销'],
            CHECK=lambda df: check_condition(df['EBITDA'] > 0),
            净债务_EBITDA=lambda df: df['净债务'] / df['EBITDA']
        )['净债务_EBITDA'],
        'formula': '净债务 / EBITDA；净债务 = 有息负债（短期借款 + 一年内到期非流动负债 + 长期借款 + 应付债券 + 租赁负债 + 应付利息 + 应付短期债券 + 交易性金融负债 + 卖出回购金融资产款）- 现金类资产（货币资金 + 交易性金融资产）；EBITDA = 利润总额 + 利息支出 + 折旧与摊销'
    },
    # 成长能力
    '营业收入增长率': {
        'sub-category': 'growth',
        'data': ['营业收入'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            营业收入增长率=lambda df: df['营业收入'].diff() / df['营业收入'].shift(1) * 100
        )['营业收入增长率'],
        'formula': '(本期营业收入 - 上期营业收入) / 上期营业收入 * 100%',
        'note': '单位：%',
    },
    'EBITDA增长率': {
        'sub-category': 'growth',
        'data': ['利润总额', '财务费用——利息费用', '固定资产折旧、油气资产折耗、生产性生物资产折旧', '无形资产摊销', '长期待摊费用摊销'],
        'function': lambda df: df.assign(
            EBITDA=lambda df: df['利润总额'] + df['财务费用——利息费用'] + df['固定资产折旧、油气资产折耗、生产性生物资产折旧'] + df['无形资产摊销'] + df['长期待摊费用摊销'],
            CHECK=lambda df: check_condition(df['EBITDA'] > 0),
            EBITDA增长率=lambda df: df['EBITDA'].diff() / df['EBITDA'].shift(1) * 100
        )['EBITDA增长率'],
        'formula': 'EBITDA增长率 = (本期EBITDA - 上期EBITDA) / 上期EBITDA * 100%；EBITDA = 利润总额 + 利息支出 + 折旧与摊销',
        'note': '单位：%',
    },
    '净利润增长率': {
        'sub-category': 'growth',
        'data': ['净利润'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['净利润'].shift(1) != 0),
            净利润增长率=lambda df: df['净利润'].diff() / df['净利润'].shift(1).abs() * 100
        )['净利润增长率'],
        'formula': '(本期净利润 - 上期净利润) / |上期净利润| * 100%',
        'note': '单位：%',
    },
    '每股收益增长率': {
        'sub-category': 'growth',
        'data': ['归属于母公司所有者的净利润', '股本'],
        'function': lambda df: df.assign(
            EPS=lambda df: df['归属于母公司所有者的净利润'] / df['股本'],
            CHECK=lambda df: check_condition(df['EPS'].shift(1) != 0),
            每股收益增长率=lambda df: df['EPS'].diff() / df['EPS'].shift(1).abs() * 100
        )['每股收益增长率'],
        'formula': '(本期每股收益 - 上期每股收益) / |上期每股收益| * 100%',
        'note': '单位：%；净利润用归母净利润'
    },
    # 现金流
    '公司自由现金流': {
        'sub-category': 'cash flow',
        'data': ['经营活动产生的现金流量净额', '购建固定资产、无形资产和其他长期资产支付的现金', '财务费用——利息费用', '所得税费用', '利润总额'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['所得税费用'] >= 0) & (df['利润总额'] > 0)),
            公司自由现金流=lambda df: df['经营活动产生的现金流量净额'] - df['购建固定资产、无形资产和其他长期资产支付的现金'] + df['财务费用——利息费用'] * (1 - df['所得税费用'] / df['利润总额'])
        )['公司自由现金流'],
        'formula': '经营活动现金流净额 - 资本支出 + 利息支出 * (1 - 所得税费用 / 利润总额)',
        'note': '单位：元'
    },
    '股权自由现金流': {
        'sub-category': 'cash flow',
        'data': ['经营活动产生的现金流量净额', '购建固定资产、无形资产和其他长期资产支付的现金', '取得借款收到的现金', '发行债券收到的现金', '偿还债务支付的现金'],
        'function': lambda df: df['经营活动产生的现金流量净额'] - df['购建固定资产、无形资产和其他长期资产支付的现金'] + (df['取得借款收到的现金'] + df['发行债券收到的现金'] - df['偿还债务支付的现金'] ),
        'formula': '经营活动现金流净额 - 资本支出 + 净借款（取得借款收到的现金 + 发行债券收到的现金 - 偿还债务支付的现金）',
        'note': '单位：元'
    },
    '经营现金流利润率': {
        'sub-category': 'cash flow',
        'data': ['经营活动产生的现金流量净额', '营业收入'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            经营现金流利润率=lambda df: df['经营活动产生的现金流量净额'] / df['营业收入'] * 100
        )['经营现金流利润率'],
        'formula': '经营活动现金流净额 / 营业收入 * 100%',
        'note': '单位：%',
    },
    '资本支出占收入比': {
        'sub-category': 'cash flow',
        'data': ['购建固定资产、无形资产和其他长期资产支付的现金', '营业收入'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['营业收入'] > 0),
            资本支出占收入比=lambda df: df['购建固定资产、无形资产和其他长期资产支付的现金'] / df['营业收入'] * 100
        )['资本支出占收入比'],
        'formula': '资本支出 / 营业收入 * 100%',
        'note': '单位：%',
    },
    '现金流量充裕率': {
        'sub-category': 'cash flow',
        'data': ['经营活动产生的现金流量净额', '购建固定资产、无形资产和其他长期资产支付的现金', '偿还债务支付的现金', '分配股利、利润或偿付利息支付的现金'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['购建固定资产、无形资产和其他长期资产支付的现金'] + df['偿还债务支付的现金'] + df['分配股利、利润或偿付利息支付的现金'] > 0),
            现金流量充裕率=df['经营活动产生的现金流量净额'] / (df['购建固定资产、无形资产和其他长期资产支付的现金'] + df['偿还债务支付的现金'] + df['分配股利、利润或偿付利息支付的现金'])
        )['现金流量充裕率'],
        'formula': '经营活动现金流净额 / (资本支出 + 偿还债务支付的现金 + 分配股利、利润或偿付利息支付的现金)'
    },
    # 每股指标
    '每股净资产': {
        'sub-category': 'per share',
        'data': ['股东权益合计', '股本'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['股本'] > 0),
            每股净资产=lambda df: df['股东权益合计'] / df['股本']
        )['每股净资产'],
        'formula': '股东权益合计 / 股本'
    },
    '每股资本公积金': {
        'sub-category': 'per share',
        'data': ['资本公积', '股本'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['股本'] > 0),
            每股资本公积金=lambda df: df['资本公积'] / df['股本']
        )['每股资本公积金'],
        'formula': '资本公积 / 股本'
    },
    '每股未分配利润': {
        'sub-category': 'per share',
        'data': ['未分配利润', '股本'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['股本'] > 0),
            每股未分配利润=lambda df: df['未分配利润'] / df['股本']
        )['每股未分配利润'],
        'formula': '未分配利润 / 股本'
    },
    '每股经营现金流': {
        'sub-category': 'per share',
        'data': ['经营活动产生的现金流量净额', '股本'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['股本'] > 0),
            每股经营现金流=lambda df: df['经营活动产生的现金流量净额'] / df['股本']
        )['每股经营现金流'],
        'formula': '经营活动现金流净额 / 股本'
    },
    # 杜邦分析
    # 净资产收益率 = 净利率 * 总资产周转率 * 权益乘数
    '权益乘数': {
        'sub-category': 'solvency',
        'data': ['资产合计', '股东权益合计'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['股东权益合计'] > 0),
            权益乘数=lambda df: (df['资产合计'].shift(1) + df['资产合计']) / (df['股东权益合计'].shift(1) + df['股东权益合计'])
        )['权益乘数'],
        'note': '用期初期末平均值',
        'formula': '平均总资产 / 平均股东权益'
    },
    # 估值指标
    'EBIT': {
        'sub-category': 'valuation',
        'data': ['利润总额', '财务费用——利息费用'],
        'function': lambda df: df['利润总额'] + df['财务费用——利息费用'],
        'formula': '利润总额 + 利息支出',
        'note': '单位：元'
    },
    'EBITDA': {
        'sub-category': 'valuation',
        'data': ['利润总额', '财务费用——利息费用', '固定资产折旧、油气资产折耗、生产性生物资产折旧', '无形资产摊销', '长期待摊费用摊销'],
        'function': lambda df: df['利润总额'] + df['财务费用——利息费用'] + df['固定资产折旧、油气资产折耗、生产性生物资产折旧'] + df['无形资产摊销'] + df['长期待摊费用摊销'],
        'formula': '利润总额 + 利息支出 + 折旧与摊销',
        'note': '单位：元'
    },
    'NOPAT': {
        'sub-category': 'valuation',
        'data': ['利润总额', '财务费用——利息费用', '所得税费用'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['所得税费用'] >= 0) & (df['利润总额'] > 0)),
            NOPAT=lambda df: (df['利润总额'] + df['财务费用——利息费用']) * (1 - df['所得税费用'] / df['利润总额'])
        )['NOPAT'],
        'formula': 'EBIT × (1 - 实际税率) = (利润总额 + 利息支出) × (1 - 所得税费用 / 利润总额)',
        'note': '单位：元'
    }
}



def _calc_KAMA(df, n_er=10, n_fast=2, n_slow=30):
    close = df['close'].values
    length = len(close)

    er = np.full(length, np.nan)
    sc = np.full(length, np.nan)
    kama = np.full(length, np.nan)

    # 固定常数
    fast_sc = 2 / (n_fast + 1)
    slow_sc = 2 / (n_slow + 1)

    # 1. Direction: n_er 期净变化
    direction = np.abs(close[n_er:] - close[:-n_er])  # 长度 = length - n_er

    # 2. Volatility: n_er 期变动总和
    volatility = np.abs(np.diff(close))
    vol_sum = np.convolve(volatility, np.ones(n_er), mode='valid')  # 长度 = length - n_er

    with np.errstate(divide='ignore', invalid='ignore'):
        er_values = np.where(vol_sum != 0, direction / vol_sum, 0.0)

    er[n_er:] = er_values

    # 3. SC
    sc[n_er:] = (er[n_er:] * (fast_sc - slow_sc) + slow_sc) ** 2

    # 4. 初始值
    kama[n_er - 1] = np.mean(close[:n_er])

    # 5. 递推
    for i in range(n_er, length):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    return pd.Series(kama, index=df.index)

def _calc_SAR(df, initial_af=0.02, step_af=0.02, max_af=0.20):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    length = len(close)

    sar = np.full(length, np.nan)
    if length < 2:
        return pd.Series(sar, index=df.index)

    # 初始化状态
    trend = 1 if close[1] > close[0] else -1
    ep = high[1] if trend == 1 else low[1]
    af = initial_af
    sar[1] = low[0] if trend == 1 else high[0]

    for i in range(2, length):
        # 1. 先算出理论上的下一个 SAR 值
        next_sar = sar[i-1] + af * (ep - sar[i-1])

        # 2. 根据当前趋势，处理翻转和状态更新
        if trend == 1:
            if low[i] < next_sar:
                # 发生翻转向下
                trend = -1
                next_sar = ep  # 旧的最高点变成新的SAR
                ep = low[i]    # 极值点重置为当前最低点
                af = initial_af
            else:
                # 上升趋势延续，更新极值点和加速因子
                if high[i] > ep:
                    ep = high[i]
                    af = min(max_af, af + step_af)

            # 3. 统一进行边界控制（无论是刚翻转，还是延续，最后都要卡一下边界）
            sar[i] = min(next_sar, low[i-1], low[i-2])

        else: # trend == -1
            if high[i] > next_sar:
                # 发生翻转向上
                trend = 1
                next_sar = ep  # 旧的最低点变成新的SAR
                ep = high[i]   # 极值点重置为当前最高点
                af = initial_af
            else:
                # 下降趋势延续，更新极值点和加速因子
                if low[i] < ep:
                    ep = low[i]
                    af = min(max_af, af + step_af)

            # 3. 统一进行边界控制
            sar[i] = max(next_sar, high[i-1], high[i-2])

    return pd.Series(sar, index=df.index)

def _calc_supertrend(df, n=10, m=3):
    '''
    标准 SuperTrend 计算 (采用 Wilder's RMA 平滑 ATR)
    输出 Middle, Upper, Lower
    '''
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    length = len(c)

    # 初始化输出数组
    middle = (h + l) / 2.0
    final_upper = np.full(length, np.nan)
    final_lower = np.full(length, np.nan)

    if length < n:
        return pd.DataFrame({'Middle': middle, 'Upper': final_upper, 'Lower': final_lower}, index=df.index)

    # 1. 计算 True Range
    tr = np.zeros(length)
    tr[0] = h[0] - l[0]
    for i in range(1, length):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    # 2. 计算 ATR (使用 Wilder 平滑 / RMA)
    atr = np.zeros(length)
    atr[n-1] = np.mean(tr[:n])  # 种子值用前 n 根的简单平均
    for i in range(n, length):
        atr[i] = (atr[i-1] * (n - 1) + tr[i]) / n

    # 3. 计算基础带
    basic_upper = middle + m * atr
    basic_lower = middle - m * atr

    # 4. 状态机循环 (核心逻辑)
    # 初始状态假设为多头 (1为多头，-1为空头)
    trend = 1
    curr_upper = basic_upper[n-1]
    curr_lower = basic_lower[n-1]

    for i in range(n, length):
        if trend == 1:
            # 多头状态：下轨只升不降
            curr_lower = max(basic_lower[i], curr_lower)

            # 翻转判定：收盘价跌破活跃下轨
            if c[i] < curr_lower:
                trend = -1
                curr_upper = basic_upper[i]  # 翻转瞬间，重置上轨
                # 注意：翻转后，curr_lower 保持不变，作为上方的压力位冻结
            # else: 多头延续，curr_upper 保持冻结不动

        else: # trend == -1
            # 空头状态：上轨只降不升
            curr_upper = min(basic_upper[i], curr_upper)

            # 翻转判定：收盘价突破活跃上轨
            if c[i] > curr_upper:
                trend = 1
                curr_lower = basic_lower[i]  # 翻转瞬间，重置下轨
                # 注意：翻转后，curr_upper 保持不变，作为下方的支撑位冻结
            # else: 空头延续，curr_lower 保持冻结不动

        # 记录最终输出的轨道值
        final_upper[i] = curr_upper
        final_lower[i] = curr_lower

    return pd.DataFrame({
        'Middle': middle,
        'Upper': final_upper,
        'Lower': final_lower
    }, index=df.index)


indicators_technical_ma = {
    '简单移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df['close'].rolling(window=n).mean(),
        'formula': r'SMA_n = 1 / n * \sum_{i=0}^{n-1} P_{t-i}'
    },
    '加权移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df['close'].rolling(window=n).apply(lambda x: (x * range(1, n + 1)).sum() / sum(range(1, n + 1)), raw=True),
        'note': '权重取 1, 2, ..., n',
        'formula': r'WMA_n = \sum_{i=0}^{n-1} (n-i) * P_{t-i} / \sum_{j=1}^{n} j'
    },
    '指数移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df['close'].ewm(span=n, min_periods=n, adjust=False).mean(),
        'note': '平滑系数 alpha 取 2 / (n + 1)；从前 4n 个数据开始计算；初始值用首个数据点',
        'data_lookback_n': lambda n: n * 4,
        'formula': r'EMA_t = \alpha * P_t + (1 - \alpha) * EMA_{t-1}，\alpha = 2 / (n + 1)'
    },
    '双指数移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df.assign(
            EMA1=lambda df: df['close'].ewm(span=n, min_periods=n, adjust=False).mean(),
            EMA2=lambda df: df['EMA1'].ewm(span=n, min_periods=n, adjust=False).mean(),
            DEMA=lambda df: 2 * df['EMA1'] - df['EMA2']
        )['DEMA'],
        'note': '平滑系数 alpha 取 2 / (n + 1)；从前 5n 个数据开始计算；初始值用首个数据点',
        'data_lookback_n': lambda n: n * 5,
        'formula': 'DEMA = 2 * EMA1 - EMA2；EMA1 = EMA_n(P)；EMA2 = EMA_n(EMA1)'
    },
    '三指数移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df.assign(
            EMA1=lambda df: df['close'].ewm(span=n, min_periods=n, adjust=False).mean(),
            EMA2=lambda df: df['EMA1'].ewm(span=n, min_periods=n, adjust=False).mean(),
            EMA3=lambda df: df['EMA2'].ewm(span=n, min_periods=n, adjust=False).mean(),
            TEMA=lambda df: 3 * df['EMA1'] - 3 * df['EMA2'] + df['EMA3']
        )['TEMA'],
        'note': '平滑系数 alpha 取 2 / (n + 1)；从前 6n 个数据开始计算；初始值用首个数据点',
        'data_lookback_n': lambda n: n * 6,
        'formula': 'TEMA = 3 * EMA1 - 3 * EMA2 + EMA3；EMA1 = EMA_n(P)；EMA2 = EMA_n(EMA1)；EMA3 = EMA_n(EMA2)'
    },
    '赫尔移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df.assign(
            WMA=lambda df: df['close'].rolling(window=n).apply(lambda x: (x * range(1, n + 1)).sum() / sum(range(1, n + 1)), raw=True),
            WMA_half=lambda df: df['close'].rolling(window=n//2).apply(lambda x: (x * range(1, n//2 + 1)).sum() / sum(range(1, n//2 + 1)), raw=True),
            HMA=lambda df: (2 * df['WMA_half'] - df['WMA']).rolling(window=int(n**0.5)).apply(lambda x: (x * range(1, int(n**0.5) + 1)).sum() / sum(range(1, int(n**0.5) + 1)), raw=True)
        )['HMA'],
        'data_lookback_n': lambda n: int(n * 1.5),
        'formula': r'HMA = WMA_{\sqrt{n}}(2 * WMA_{n/2}(P) - WMA_n(P))'
    },
}

indicators_technical_channel = {
    '布林带': {
        'sub-category': 'channel',
        'data': ['close'],
        'function': lambda df, n=20: df.assign(
            Middle=lambda df: df['close'].rolling(window=n).mean(),
            STD=lambda df: df['close'].rolling(window=n).std(ddof=0),
            Upper=lambda df: df['Middle'] + 2 * df['STD'],
            Lower=lambda df: df['Middle'] - 2 * df['STD']
        )[['Upper', 'Middle', 'Lower']],
        'note': '标准版：中轨SMA(20)，乘数2，用总体标准差',
        'formula': r'Middle = SMA_n(P)；Upper = Middle + 2 * \sigma_n(P)；Lower = Middle - 2 * \sigma_n(P)'
    },
    '肯特纳通道': {
        'sub-category': 'channel',
        'data': ['close', 'high', 'low'],
        'function': lambda df, n_ema=20, n_atr=10, multiplier=2: df.assign(
            Middle=lambda df: df['close'].ewm(span=n_ema, min_periods=n_ema, adjust=False).mean(),
            ATR=lambda df: pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1).rolling(window=n_atr).mean(),
            Upper=lambda df: df['Middle'] + multiplier * df['ATR'],
            Lower=lambda df: df['Middle'] - multiplier * df['ATR']
        )[['Upper', 'Middle', 'Lower']],
        'note': '标准版：中轨EMA(20)，ATR(10)，乘数2；EMA从前4n个数据开始计算，初始值用首个数据点；ATR计算用SMA',
        'data_lookback_n': lambda n_ema=20, **kwargs: n_ema * 4,
        'formula': 'Middle = EMA_{n_{ema}}(C)；Upper = Middle + 2 * ATR_{n_{atr}}；Lower = Middle - 2 * ATR_{n_{atr}}'
    },
    '包络线': {
        'sub-category': 'channel',
        'data': ['close'],
        'function': lambda df, n=20: df.assign(
            Middle=lambda df: df['close'].rolling(window=n).mean(),
            Upper=lambda df: df['Middle'] * 1.1,
            Lower=lambda df: df['Middle'] * 0.9
        )[['Upper', 'Middle', 'Lower']],
        'note': '标准版：中轨SMA(20)，上下轨±10%',
        'formula': 'Middle = SMA_n(P)；Upper = Middle * 1.1；Lower = Middle * 0.9'
    },
    '唐安奇通道': {
        'sub-category': 'channel',
        'data': ['high', 'low'],
        'function': lambda df, n=20: df.assign(
            Upper=lambda df: df['high'].rolling(window=n).max(),
            Lower=lambda df: df['low'].rolling(window=n).min(),
            Middle=lambda df: (df['Upper'] + df['Lower']) / 2
        )[['Upper', 'Middle', 'Lower']],
        'note': '标准版：20日内最高价和最低价，中轨用两者均值',
        'formula': 'Upper = max_{i=0}^{n-1} H_{t-i}；Lower = min_{i=0}^{n-1} L_{t-i}；Middle = (Upper + Lower) / 2'
    },
    '超级趋势': {
        'sub-category': 'channel',
        'data': ['high', 'low', 'close'],
        'function': _calc_supertrend,
        'note': '标准版：周期10，乘数3；回看前5n个数据',
        'data_lookback_n': lambda n=10, **kwargs: n * 5,
        'formula': 'Mid_t = (H_t + L_t) / 2；BasicUpper_t = Mid_t + m * ATR_n^{Wilder}；BasicLower_t = Mid_t - m * ATR_n^{Wilder}；多头时 Lower_t = max(BasicLower_t, Lower_{t-1})，若 C_t < Lower_t 则翻空；空头时 Upper_t = min(BasicUpper_t, Upper_{t-1})，若 C_t > Upper_t 则翻多'
    },
}

indicators_technical_n = {
    'Aroon上升': {
        'sub-category': 'trend',
        'data': ['high'],
        'function': lambda df, n=25: df.assign(
            AroonUp=lambda df: df['high'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmax()) / n * 100, raw=True),
        )['AroonUp'],
        'note': '标准版：周期25',
        'data_lookback_n': lambda n=25: n,
        'formula': 'AroonUp = (n - i_{max}) / n * 100；i_{max} = argmax_{i=0}^{n} H_{t-i}，为 n+1 期内最高价距今天数'
    },
    'Aroon下降': {
        'sub-category': 'trend',
        'data': ['high', 'low'],
        'function': lambda df, n=25: df.assign(
            AroonDown=lambda df: df['low'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmin()) / n * 100, raw=True),
        )['AroonDown'],
        'note': '标准版：周期25',
        'data_lookback_n': lambda n=25: n,
        'formula': 'AroonDown = (n - i_{min}) / n * 100；i_{min} = argmin_{i=0}^{n} L_{t-i}，为 n+1 期内最低价距今天数'
    },
    'Aroon振荡': {
        'sub-category': 'trend',
        'data': ['high', 'low'],
        'function': lambda df, n=25: df.assign(
            AroonUp=lambda df: df['high'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmax()) / n * 100, raw=True),
            AroonDown=lambda df: df['low'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmin()) / n * 100, raw=True),
            AroonOsc=lambda df: df['AroonUp'] - df['AroonDown']
        )['AroonOsc'],
        'note': '标准版：周期25',
        'data_lookback_n': lambda n=25: n,
        'formula': 'AroonOsc = AroonUp - AroonDown；AroonUp = (n - i_{max}) / n * 100；i_{max} = argmax_{i=0}^{n} H_{t-i}，为 n+1 期内最高价距今天数；AroonDown = (n - i_{min}) / n * 100；i_{min} = argmin_{i=0}^{n} L_{t-i}，为 n+1 期内最低价距今天数'
    },
    '相对强弱指标': {
        'sub-category': 'momentum',
        'data': ['close'],
        'function': lambda df, n=14: df.assign(
            Change=lambda df: df['close'].diff(),
            Gain=lambda df: df['Change'].clip(lower=0).rolling(window=n).mean(),
            Loss=lambda df: -df['Change'].clip(upper=0).rolling(window=n).mean(),
            RS=lambda df: df['Gain'] / df['Loss'],
            RSI=lambda df: 100 - (100 / (1 + df['RS']))
        )['RSI'],
        'note': '标准版：周期14；Gain/Loss计算用SMA',
        'data_lookback_n': lambda n=14: n,
        'formula': 'U_t = max(P_t - P_{t-1}, 0)；D_t = max(P_{t-1} - P_t, 0)；RS = SMA_n(U) / SMA_n(D)；RSI = 100 - 100 / (1 + RS)'
    },
    '商品通道指数': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=20: df.assign(
            TP=lambda df: (df['high'] + df['low'] + df['close']) / 3,
            SMA_TP=lambda df: df['TP'].rolling(window=n).mean(),
            MeanDev=lambda df: df['TP'].rolling(window=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True),
            CCI=lambda df: (df['TP'] - df['SMA_TP']) / (0.015 * df['MeanDev'])
        )['CCI'],
        'note': '标准版：周期20',
        'formula': r'TP_t = (H_t + L_t + C_t) / 3；MD_t = 1 / n * \sum_{i=0}^{n-1} |TP_{t-i} - SMA_n(TP)|；CCI = (TP_t - SMA_n(TP)) / (0.015 * MD_t)'
    },
    '威廉%R': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=14: df.assign(
            Highest=lambda df: df['high'].rolling(window=n).max(),
            Lowest=lambda df: df['low'].rolling(window=n).min(),
            WilliamsR=lambda df: (df['Highest'] - df['close']) / (df['Highest'] - df['Lowest']) * (-100)
        )['WilliamsR'],
        'note': '标准周期14',
        'formula': r'%R = (max_{i=0}^{n-1} H_{t-i} - C_t) / (max_{i=0}^{n-1} H_{t-i} - min_{i=0}^{n-1} L_{t-i}) * (-100)'
    },
    '布林带%B': {
        'sub-category': 'volatility',
        'data': ['close'],
        'function': lambda df, n=20: df.assign(
            Middle=lambda df: df['close'].rolling(window=n).mean(),
            STD=lambda df: df['close'].rolling(window=n).std(ddof=0),
            Upper=lambda df: df['Middle'] + 2 * df['STD'],
            Lower=lambda df: df['Middle'] - 2 * df['STD'],
            PctB=lambda df: (df['close'] - df['Lower']) / (df['Upper'] - df['Lower'])
        )['PctB'],
        'note': '标准版：周期20，乘数2，用总体标准差',
        'formula': r'Middle = SMA_n(P)；Upper = Middle + 2 * \sigma_n(P)；Lower = Middle - 2 * \sigma_n(P)；%B = (P_t - Lower) / (Upper - Lower)'
    },
    '平均真实波幅': {
        'sub-category': 'volatility',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=14: pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1).rolling(window=n).mean(),
        'note': '标准版：周期14',
        'data_lookback_n': lambda n=14: n,
        'formula': 'TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)；ATR = SMA_n(TR)'
    },
    '易动指标': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'volume'],
        'function': lambda df, n=14: df.assign(
            HL_Range=lambda df: np.where(df['high'] == df['low'], 1e-10, df['high'] - df['low']),
            MidMove=lambda df: ((df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2) / ((df['volume'] / 10**6) / df['HL_Range']),
            EMV=lambda df: df['MidMove'].rolling(window=n).mean()
        )['EMV'],
        'note': '标准版：周期14，分母中 volume 以百万为单位，避免数值过大',
        'data_lookback_n': lambda n=14: n,
        'formula': 'MidMove_t = ((H_t + L_t) / 2 - (H_{t-1} + L_{t-1}) / 2) / ((V_t / 10^6) / (H_t - L_t))；EMV = SMA_n(MidMove)'
    },
    '资金流量指数': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df, n=14: df.assign(
            TP=lambda df: (df['high'] + df['low'] + df['close']) / 3,
            MF=lambda df: df['TP'] * df['volume'],
            PosMF=lambda df: np.where(df['TP'] > df['TP'].shift(1), df['MF'], 0) * np.where(df['TP'].shift(1).isna(), np.nan, 1),  # 第一个数据点为nan
            NegMF=lambda df: np.where(df['TP'] < df['TP'].shift(1), df['MF'], 0) * np.where(df['TP'].shift(1).isna(), np.nan, 1),  # 第一个数据点为nan
            PosSum=lambda df: df['PosMF'].rolling(window=n).sum(),
            NegSum=lambda df: df['NegMF'].rolling(window=n).sum(),
            MFI=lambda df: 100 - (100 / (1 + df['PosSum'] / df['NegSum'].replace(0, np.nan)))
        )['MFI'],
        'note': '标准版：周期14',
        'data_lookback_n': lambda n=14: n,
        'formula': r'TP_t = (H_t + L_t + C_t) / 3；MF_t = TP_t * V_t；PMF_t = MF_t * \mathbb{1}[TP_t > TP_{t-1}]；NMF_t = MF_t * \mathbb{1}[TP_t < TP_{t-1}]；MFR = \sum_{i=0}^{n-1} PMF_{t-i} / \sum_{i=0}^{n-1} NMF_{t-i}；MFI = 100 - 100 / (1 + MFR)'
    },
    'KDJ的K': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=9: df.assign(
            RSV=lambda df: (df['close'] - df['low'].rolling(window=n).min()) / (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()) * 100,
            K=lambda df: df['RSV'].ewm(com=2, min_periods=5, adjust=False).mean(),
        )['K'],
        'note': '标准版：RSV周期n=9，1/3平滑权重；从前20+n个数据开始计算，递推初始值用首个数据点',
        'data_lookback_n': lambda n=9: 20 + n,
        'formula': 'K_t = 1/3 * RSV_t + 2/3 * K_{t-1}；RSV_t = (C_t - min_{i=0}^{n-1} L_{t-i}) / (max_{i=0}^{n-1} H_{t-i} - min_{i=0}^{n-1} L_{t-i}) * 100'
    },
    'KDJ的D': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=9: df.assign(
            RSV=lambda df: (df['close'] - df['low'].rolling(window=n).min()) / (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()) * 100,
            K=lambda df: df['RSV'].ewm(com=2, min_periods=5, adjust=False).mean(),
            D=lambda df: df['K'].ewm(com=2, min_periods=5, adjust=False).mean(),
        )['D'],
        'note': '标准版：RSV周期n=9，1/3平滑权重；从前25+n个数据开始计算，递推初始值用首个数据点',
        'data_lookback_n': lambda n=9: 25 + n,
        'formula': 'D_t = 1/3 * K_t + 2/3 * D_{t-1}；K_t = 1/3 * RSV_t + 2/3 * K_{t-1}；RSV_t = (C_t - min_{i=0}^{n-1} L_{t-i}) / (max_{i=0}^{n-1} H_{t-i} - min_{i=0}^{n-1} L_{t-i}) * 100'
    },
    'KDJ的J': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=9: df.assign(
            RSV=lambda df: (df['close'] - df['low'].rolling(window=n).min()) / (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()) * 100,
            K=lambda df: df['RSV'].ewm(com=2, min_periods=5, adjust=False).mean(),
            D=lambda df: df['K'].ewm(com=2, min_periods=5, adjust=False).mean(),
            J=lambda df: 3 * df['K'] - 2 * df['D']
        )['J'],
        'note': '标准版：RSV周期n=9，1/3平滑权重；从前25+n个数据开始计算，递推初始值用首个数据点',
        'data_lookback_n': lambda n=9: 25 + n,
        'formula': 'J_t = 3 * K_t - 2 * D_t；D_t = 1/3 * K_t + 2/3 * D_{t-1}；K_t = 1/3 * RSV_t + 2/3 * K_{t-1}；RSV_t = (C_t - min_{i=0}^{n-1} L_{t-i}) / (max_{i=0}^{n-1} H_{t-i} - min_{i=0}^{n-1} L_{t-i}) * 100'
    },
}

indicators_technical_other = {
    '考夫曼自适应移动平均': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': _calc_KAMA,
        'note': 'ER周期取10，快线周期取2，慢线周期取30；从前60个数据开始计算',
        'data_lookback_n': lambda n_slow=30, **kwargs: n_slow * 2,
        'formula': r'ER_t = |C_t - C_{t-n_{er}}| / \sum_{i=0}^{n_{er}-1} |C_{t-i} - C_{t-i-1}|；SC_t = (ER_t * (2/(n_{fast}+1) - 2/(n_{slow}+1)) + 2/(n_{slow}+1))^2；KAMA_t = KAMA_{t-1} + SC_t * (C_t - KAMA_{t-1})'
    },
    'MACD的DIF': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            DIF=lambda df: df['EMA_fast'] - df['EMA_slow'],
        )['DIF'],
        'note': '标准版：快线12，慢线26；从前104个数据开始计算，EMA初始值用首个数据点',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'DIF = EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)'
    },
    'MACD的DEA': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26, n_signal=9: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            DIF=lambda df: df['EMA_fast'] - df['EMA_slow'],
            DEA=lambda df: df['DIF'].ewm(span=n_signal, min_periods=n_signal, adjust=False).mean(),
        ),
        'note': '标准版：快线12，慢线26，信号线9；从前104个数据开始计算，EMA初始值用首个数据点',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'DIF = EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)；DEA = EMA_{n_{signal}}(DIF)'
    },
    'MACD': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26, n_signal=9: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            DIF=lambda df: df['EMA_fast'] - df['EMA_slow'],
            DEA=lambda df: df['DIF'].ewm(span=n_signal, min_periods=n_signal, adjust=False).mean(),
            MACD=lambda df: (df['DIF'] - df['DEA']) * 2
        )['MACD'],
        'note': '标准版：快线12，慢线26，信号线9，A股MACD柱乘2；从前104个数据开始计算，EMA初始值用首个数据点',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'DIF = EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)；DEA = EMA_{n_{signal}}(DIF)；MACD = (DIF - DEA) * 2'
    },
    '抛物线SAR': {
        'sub-category': 'trend',
        'data': ['high', 'low', 'close'],
        'function': _calc_SAR,
        'note': '初始alpha=0.02，步长0.02，最大值0.20；从前10个数据开始计算',
        'data_lookback_n': lambda **kwargs: 10,
        'formula': 'SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1})；EP 为当前趋势的极值点（多头取最高、空头取最低）；AF 从 initial_af 起步，每次刷新极值累加 step_af 至 max_af；趋势翻转时 SAR 取上一段 EP、AF 重置为 initial_af'
    },
    '随机RSI': {
        'sub-category': 'momentum',
        'data': ['close'],
        'function': lambda df, n=14, k=14: df.assign(
            Change=lambda df: df['close'].diff(),
            Gain=lambda df: df['Change'].clip(lower=0).rolling(window=n).mean(),
            Loss=lambda df: -df['Change'].clip(upper=0).rolling(window=n).mean(),
            RSI=lambda df: 100 - (100 / (1 + (df['Gain'] / df['Loss']))),
            MinRSI=lambda df: df['RSI'].rolling(window=k).min(),
            MaxRSI=lambda df: df['RSI'].rolling(window=k).max(),
            StochRSI=lambda df: (df['RSI'] - df['MinRSI']) / (df['MaxRSI'] - df['MinRSI'])
        )['StochRSI'],
        'note': '标准版：周期14；从前27个数据开始计算',
        'data_lookback_n': lambda n=14, k=14: n + k - 1,
        'formula': 'RSI_t = 100 - 100 / (1 + SMA(Gain, n) / SMA(Loss, n))；StochRSI_t = (RSI_t - min_{i=0}^{k-1} RSI_{t-i}) / (max_{i=0}^{k-1} RSI_{t-i} - min_{i=0}^{k-1} RSI_{t-i})'
    },
    '百分比价格振荡': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            PPO=lambda df: (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow'] * 100
        )['PPO'],
        'note': '标准版：快线12，慢线26；从前104个数据开始计算，EMA初始值用首个数据点',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'PPO = (EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)) / EMA_{n_{slow}}(P) * 100'
    },
    '真实波幅': {
        'sub-category': 'volatility',
        'data': ['high', 'low', 'close'],
        'function': lambda df: pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1),
        'data_lookback_n': lambda **kwargs: 1,
        'formula': 'TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)'
    },
    '蔡金波动率': {
        'sub-category': 'volatility',
        'data': ['high', 'low'],
        'function': lambda df, n=10, m=10: df.assign(
            HL_EMA=lambda df: (df['high'] - df['low']).ewm(span=n, min_periods=n, adjust=False).mean(),
            ChaikinVol=lambda df: (df['HL_EMA'] / df['HL_EMA'].shift(m)) - 1
        )['ChaikinVol'],
        'note': '标准版：参数 (10, 10)；从前50个数据开始计算，EMA初始值用首个数据点',
        'data_lookback_n': lambda n=10, m=10: n * 4 + m,
        'formula': 'HL_EMA_t = EMA_n(H - L)；ChaikinVol_t = HL_EMA_t / HL_EMA_{t-m} - 1'
    },
    '成交量加权平均价': {
        'sub-category': 'volume',
        'data': ['amount', 'volume'],
        'function': lambda df: df['amount'] / df['volume'] ,
        'note': '直接使用 amount 计算',
        'formula': 'VWAP_t = amount_t / V_t'
    },
    '能量潮': {
        'sub-category': 'volume',
        'data': ['close', 'volume'],
        'function': lambda df: (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum(),
        'note': '从前60个数据开始计算',
        'data_lookback_n': lambda **kwargs: 60,
        'formula': r'OBV_t = OBV_{t-1} + sign(C_t - C_{t-1}) * V_t = \sum_{i=1}^{t} sign(C_i - C_{i-1}) * V_i'
    },
    '力度指数': {
        'sub-category': 'volume',
        'data': ['close', 'volume'],
        'function': lambda df: df['close'].diff() * df['volume'],
        'data_lookback_n': lambda **kwargs: 1,
        'formula': 'ForceIndex_t = (C_t - C_{t-1}) * V_t'
    },
    '累积/派发线': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df: df.assign(
            MFM=lambda df: ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan),
            MFV=lambda df: df['MFM'] * df['volume'],
            ADL=lambda df: df['MFV'].cumsum()
        )['ADL'],
        'note': '从前60个数据开始计算',
        'data_lookback_n': lambda **kwargs: 60,
        'formula': r'MFM_t = ((C_t - L_t) - (H_t - C_t)) / (H_t - L_t)；MFV_t = MFM_t * V_t；ADL_t = ADL_{t-1} + MFV_t = \sum_{i=1}^{t} MFV_i'
    },
    '蔡金资金流': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df, n=20: df.assign(
            MFM=lambda df: ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan),
            MFV=lambda df: df['MFM'] * df['volume'],
            CMF=lambda df: df['MFV'].rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
        )['CMF'],
        'note': '标准版：周期20',
        'data_lookback_n': lambda n=20, **kwargs: n - 1,
        'formula': r'MFM_t = ((C_t - L_t) - (H_t - C_t)) / (H_t - L_t)；MFV_t = MFM_t * V_t；CMF = \sum_{i=0}^{n-1} MFV_{t-i} / \sum_{i=0}^{n-1} V_{t-i}'
    },
    '蔡金振荡指标': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df, n_fast=3, n_slow=10: df.assign(
            MFM=lambda df: ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan),
            MFV=lambda df: df['MFM'] * df['volume'],
            ADL=lambda df: df['MFV'].cumsum(),
            ChaikinOsc=lambda df: df['ADL'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean() - df['ADL'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean()
        )['ChaikinOsc'],
        'note': '标准版：快线3，慢线10；从前60个数据开始计算，EMA初始值用首个数据点',
        'data_lookback_n': lambda **kwargs: 60,
        'formula': 'ChaikinOsc = EMA_{n_{fast}}(ADL) - EMA_{n_{slow}}(ADL)'
    },
}

indicators_technical_performance = {
    '年化收益率': {
        'sub-category': 'return',
        'data': ['close'],
        'function': lambda df: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) * 100,
        'note': 'A股按一年242个交易日；单位：%',
        'formula': '((P_T / P_0)^{242 / N} - 1) * 100%'
    },
    '年化波动率': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: df['close'].pct_change().dropna().std() * np.sqrt(242) * 100,
        'note': 'A股按一年242个交易日；单位：%',
        'formula': r'r_t = P_t / P_{t-1} - 1；AnnVol = \sigma(r) * \sqrt{242} * 100%'
    },
    '偏度': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: df['close'].pct_change().dropna().skew(),
        'formula': r'Skew(r) = E[(r - \mu)^3] / \sigma^3'
    },
    '超额峰度': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: df['close'].pct_change().dropna().kurtosis(),
        'formula': r'Kurt(r) = E[(r - \mu)^4] / \sigma^4 - 3'
    },
    '下行偏差': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, rf=0: np.sqrt(np.mean(np.minimum(df['close'].pct_change().dropna() - rf, 0)**2)),
        'note': '默认目标收益率为0；无需年化',
        'formula': r'DD = \sqrt{1 / N * \sum_{t=1}^{N} min(r_t - rf, 0)^2}'
    },
    '最大回撤': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: (1 - df['close'] / df['close'].cummax()).max() * 100,
        'note': '取绝对值；单位：%',
        'formula': r'MDD = max_{t} (1 - P_t / max_{s \le t} P_s) * 100%'
    },
    '跟踪误差': {
        'sub-category': 'risk',
        'data': ['close', 'benchmark_close'],
        'function': lambda df: (df['close'].pct_change() - df['benchmark_close'].pct_change()).dropna().std() * np.sqrt(242),
        'note': '需年化，A股按一年242个交易日',
        'formula': r'TE = \sigma(r - r_b) * \sqrt{242}；r_b 为基准日收益率'
    },
    '信息比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'benchmark_close'],
        'function': lambda df: (((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) - ((df['benchmark_close'].iloc[-1] / df['benchmark_close'].iloc[0]) ** (242 / (len(df) - 1)) - 1)) / (df['close'].pct_change() - df['benchmark_close'].pct_change()).dropna().std() / np.sqrt(242),
        'note': '需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': r'IR = (AnnRet - AnnRet_b) / TE；TE = \sigma(r - r_b) * \sqrt{242}；r_b 为基准日收益率'
    },
    '夏普比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df, rf=0: (((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) - rf) / df['close'].pct_change().dropna().std() / np.sqrt(242),
        'note': '默认无风险利率为0；需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': r'SR = (AnnRet - rf) / (\sigma(r) * \sqrt{242})'
    },
    '索提诺比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df, rf=0: (((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) - rf) / np.sqrt(np.mean(np.minimum(df['close'].pct_change().dropna() - rf, 0)**2) * 242),
        'note': '默认目标收益率为0；需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': r'Sortino = (AnnRet - rf) / (DD * \sqrt{242})'
    },
    '卡玛比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) / (1 - df['close'] / df['close'].cummax()).max(),
        'note': '收益率需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': 'Calmar = AnnRet / MDD'
    },
    '特雷诺比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'market_close'],
        'function': lambda df, rf=0: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1 - rf) / df['close'].pct_change().dropna().cov(df['market_close'].pct_change().dropna()) * df['market_close'].pct_change().dropna().var(),
        'note': '默认无风险利率为0；收益率需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': r'Treynor = (AnnRet - rf) / \beta；\beta = cov(r, r_m) / var(r_m)'
    },
    '莫迪利安尼比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'benchmark_close'],
        'function': lambda df, rf=0: rf + (((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) - rf) / df['close'].pct_change().dropna().std() * df['benchmark_close'].pct_change().dropna().std(),
        'note': '默认无风险利率为0；收益率需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': r'M^2 = rf + (AnnRet - rf) / \sigma(r) * \sigma(r_b)'
    },
    '溃疡指数': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: ((1 - df['close'] / df['close'].cummax()) ** 2).mean() ** 0.5,
        'formula': r'UI = \sqrt{1 / N * \sum_{t=1}^{N} (1 - P_t / max_{s \le t} P_s)^2}'
    },
    '马丁比率': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1) / (((1 - df['close'] / df['close'].cummax()) ** 2).mean() * 242) ** 0.5 ,
        'note': '分子分母都需年化，A股按一年242个交易日',
        'formula': r'Martin = AnnRet / (UI * \sqrt{242})'
    },
    '贝塔系数': {
        'sub-category': 'risk',
        'data': ['close', 'market_close'],
        'function': lambda df: df['close'].pct_change().dropna().cov(df['market_close'].pct_change().dropna()) / df['market_close'].pct_change().dropna().var(),
        'formula': r'\beta = cov(r, r_m) / var(r_m)'
    },
    '詹森阿尔法': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'market_close'],
        'function': lambda df, rf=0: (df['close'].iloc[-1] / df['close'].iloc[0]) ** (242 / (len(df) - 1)) - 1 - (rf + (df['close'].pct_change().dropna().cov(df['market_close'].pct_change().dropna()) / df['market_close'].pct_change().dropna().var()) * ((df['market_close'].iloc[-1] / df['market_close'].iloc[0]) ** (242 / (len(df) - 1)) - 1 - rf)),
        'note': '默认无风险利率为0；收益率需年化（用复合年化收益率），A股按一年242个交易日',
        'formula': r'\alpha = AnnRet - [rf + \beta * (AnnRet_m - rf)]；\beta = cov(r, r_m) / var(r_m)'
    },
    'VaR（参数法）': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: (
            lambda ret: norm.ppf(confidence) * ret.std() - ret.mean()
        )(df['close'].pct_change().dropna()),
        'note': '1日VaR；损失用正数表示；正态假设；默认95%置信度',
        'formula': r'VaR_c = \Phi^{-1}(c) * \sigma(r) - \mu(r)'
    },
    'VaR（历史模拟法）': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: -df['close'].pct_change().dropna().quantile(1 - confidence),
        'note': '1日VaR；损失用正数表示；默认95%置信度',
        'formula': 'VaR_c = -Q_{1-c}(r)；Q_p 为 p 分位数'
    },
    'CVaR（参数法）': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: (
            lambda ret: norm.pdf(norm.ppf(confidence)) / (1 - confidence) * ret.std() - ret.mean()
        )(df['close'].pct_change().dropna()),
        'note': '1日VaR；正态假设；默认95%置信度',
        'formula': r'CVaR_c = \phi(\Phi^{-1}(c)) / (1 - c) * \sigma(r) - \mu(r)'
    },
    'CVaR（历史模拟法）': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: (
            lambda ret: -ret[ret <= ret.quantile(1 - confidence)].mean()
        )(df['close'].pct_change().dropna()),
        'note': '1日VaR；默认95%置信度',
        'formula': r'CVaR_c = -E[r | r \le Q_{1-c}(r)]'
    },
}



indicators_macro = {
    "GDP平减指数": {
        'sub-category': 'inflation',
        'data': ['国内生产总值 (亿元)', '国内生产总值指数 (1978年=100)'],
        'function': lambda df: df.assign(
            实际GDP=lambda df: df.loc[1978, '国内生产总值 (亿元)'] * df['国内生产总值指数 (1978年=100)'] / 100,
            GDP平减指数=lambda df: df['国内生产总值 (亿元)'] / df['实际GDP'] * 100
        )['GDP平减指数'],
        'formula': 'GDP平减指数 = 名义GDP / 实际GDP * 100；实际GDP = 1978年GDP × t年GDP指数(1978年=100) / 100',
        'note': '实际GDP以1978年为基期'
    },
    '增量资本产出率': {
        'sub-category': 'productivity',
        'data': ['固定资本形成总额 (亿元)', '国内生产总值 (亿元)'],
        'function': lambda df: df.assign(
            GDP增量=lambda df: df['国内生产总值 (亿元)'].diff(),
            增量资本产出率=lambda df: df['固定资本形成总额 (亿元)'] / df['GDP增量']
        )['增量资本产出率'],
        'formula': r'增量资本产出率 = \Delta K / \Delta Y = 固定资本形成总额 / GDP增量',
        'note': '名义口径'
    },
    '劳动生产率': {
        'sub-category': 'productivity',
        'data': ['国内生产总值 (亿元)', '国内生产总值指数 (1978年=100)', '就业人员 (万人)'],
        'function': lambda df: df.assign(
            劳动生产率=lambda df: (df.loc[1978, '国内生产总值 (亿元)'] * df['国内生产总值指数 (1978年=100)'] / 100) * 10**8 / (df['就业人员 (万人)'] * 10**4)
        )['劳动生产率'],
        'formula': '实际GDP / 从业人数；实际GDP = 1978年GDP × t年GDP指数(1978年=100) / 100',
        'note': '单位：元/人；实际GDP以1978年为基期'
    },
    '全要素生产率增长率': {
        'sub-category': 'productivity',
        'data': ['国内生产总值 (亿元)', 'GDP增长率 (%)', '就业人员 (万人)', '非金融交易资金来源劳动者报酬 (亿元)'],
        'function': lambda df: df.assign(
            Delta_Y=lambda df: df['GDP增长率 (%)'] / 100,
            Delta_L=lambda df: df['就业人员 (万人)'].pct_change(),
            beta=lambda df: df['非金融交易资金来源劳动者报酬 (亿元)'] / df['国内生产总值 (亿元)'],
            alpha=lambda df: 1 - df['beta'],
            Delta_TFP=lambda df: (df['Delta_Y'] - df['alpha'] * 0.06 - df['beta'] * df['Delta_L']) * 100,
         )['Delta_TFP'],
        'formula': r'全要素生产率增长率 = \Delta TFP * 100% = (\Delta Y - \alpha \Delta K - \beta \Delta L) * 100%；\alpha = 1 - \beta；\beta = 劳动者报酬 / GDP',
        'note': r'单位：%；\Delta K 直接取 6%'
    },
    '贸易开放度': {
        'sub-category': 'external trade',
        'data': ['进出口总额 (人民币) (亿元)', '国内生产总值 (亿元)'],
        'function': lambda df: df.assign(
            贸易开放度=lambda df: df['进出口总额 (人民币) (亿元)'] / df['国内生产总值 (亿元)'] * 100
        )['贸易开放度'],
        'formula': '进出口总额 / GDP * 100%',
        'note': '单位：%',
    },
    '中央政府杠杆率': {
        'sub-category': 'fiscal',
        'data': ['中央财政债务余额 (亿元)', '国内生产总值 (亿元)'],
        'function': lambda df: df.assign(
            中央政府杠杆率=lambda df: df['中央财政债务余额 (亿元)'] / df['国内生产总值 (亿元)'] * 100
        )['中央政府杠杆率'],
        'formula': '中央财政债务余额 / GDP * 100%',
        'note': '单位：%',
    },
    '财政赤字率': {
        'sub-category': 'fiscal',
        'data': ['一般公共预算支出 (亿元)', '一般公共预算收入 (亿元)', '国内生产总值 (亿元)'],
        'function': lambda df: df.assign(
            财政赤字率=lambda df: (df['一般公共预算支出 (亿元)'] - df['一般公共预算收入 (亿元)']) / df['国内生产总值 (亿元)'] * 100
        )['财政赤字率'],
        'formula': '财政赤字 / GDP * 100%；财政赤字 = 一般公共预算支出 - 一般公共预算收入',
        'note': '单位：%',
    },
    '赤字依存度': {
        'sub-category': 'fiscal',
        'data': ['一般公共预算支出 (亿元)', '一般公共预算收入 (亿元)'],
        'function': lambda df: df.assign(
            赤字依存度=lambda df: (df['一般公共预算支出 (亿元)'] - df['一般公共预算收入 (亿元)']) / df['一般公共预算支出 (亿元)'] * 100
        )['赤字依存度'],
        'formula': '(一般公共预算支出 - 一般公共预算收入) / 一般公共预算支出 * 100%',
        'note': '单位：%',
    },
    '宏观税负': {
        'sub-category': 'fiscal',
        'data': ['税收 (亿元)', '国内生产总值 (亿元)'],
        'function': lambda df: df.assign(
            宏观税负=lambda df: df['税收 (亿元)'] / df['国内生产总值 (亿元)'] * 100
        )['宏观税负'],
        'formula': '税收 / GDP * 100%',
        'note': '单位：%',
    },
    # '产业结构高级化指数': {
    #     'sub-category': 'structure',
    #     'data': ['第一产业占GDP比重 (%)', '第二产业占GDP比重 (%)', '第三产业占GDP比重 (%)'],
    #     'function': lambda df: (df['第一产业占GDP比重 (%)'] * 1 + df['第二产业占GDP比重 (%)'] * 2 + df['第三产业占GDP比重 (%)'] * 3) / 100,
    #     'formula': '第一产业占GDP比重 * 1 + 第二产业占GDP比重 * 2 + 第三产业占GDP比重 * 3'
    # },
    # '二元经济结构对比系数': {
    #     'sub-category': 'structure',
    #     'data': ['第一产业占GDP比重 (%)', '就业人员 (万人)', '第一产业就业人员 (万人)'],
    #     'function': lambda df: df.assign(
    #         农业就业比重=lambda df: df['第一产业就业人员 (万人)'] / df['就业人员 (万人)'],
    #         二元经济结构对比系数=lambda df: df['第一产业占GDP比重 (%)'] * (1 - df['农业就业比重']) / ((100 - df['第一产业占GDP比重 (%)']) * df['农业就业比重'])
    #     )['二元经济结构对比系数'],
    #     'formula': '(农业GDP比重 / 非农业GDP比重) / (农业就业比重 / 非农业就业比重)',
    #     'note': '农业GDP比重 用 第一产业GDP比重 表示，农业就业比重 用 第一产业就业比重 表示',
    # },
    # '城乡居民收入比': {
    #     'sub-category': 'structure',
    #     'data': ['城镇居民人均可支配收入 (元)', '农村居民人均可支配收入 (元)'],
    #     'function': lambda df: df.assign(
    #         城乡居民收入比=lambda df: df['城镇居民人均可支配收入 (元)'] / df['农村居民人均可支配收入 (元)']
    #     )['城乡居民收入比'],
    #     'formula': '城镇居民人均可支配收入 / 农村居民人均可支配收入'
    # },
    '企业贷款实际利率': {
        'sub-category': 'inflation',
        'data': ['1年期LPR (%)', '工业生产者出厂价格指数 (上年=100)'],
        'function': lambda df: df['1年期LPR (%)'] - (df['工业生产者出厂价格指数 (上年=100)'] - 100),
        'formula': '名义利率 - 通胀率 = 1年期LPR - PPI同比增速',
        'note': '单位：%；LPR用每年8月份公布的1年期LPR'
    },
    '（城镇非私营单位）实际工资增长率': {
        'sub-category': 'labor',
        'data': ['城镇非私营单位就业人员平均工资 (元)', '居民消费价格指数 (上年=100)'],
        'function': lambda df: df['城镇非私营单位就业人员平均工资 (元)'].pct_change() * 100 - (df['居民消费价格指数 (上年=100)'] - 100),
        'formula': '名义工资增长率 - 通胀率 = 城镇非私营单位就业人员平均工资同比增速 - CPI同比增速',
        'note': '单位：%',
    },
    '（城镇私营单位）实际工资增长率': {
        'sub-category': 'labor',
        'data': ['城镇私营单位就业人员平均工资 (元)', '居民消费价格指数 (上年=100)'],
        'function': lambda df: df['城镇私营单位就业人员平均工资 (元)'].pct_change() * 100 - (df['居民消费价格指数 (上年=100)'] - 100),
        'formula': '名义工资增长率 - 通胀率 = 城镇私营单位就业人员平均工资同比增速 - CPI同比增速',
        'note': '单位：%',
    },
    '就业弹性': {
        'sub-category': 'labor',
        'data': ['就业人员 (万人)', 'GDP增长率 (%)'],
        'function': lambda df: df.assign(
            就业弹性=lambda df: df['就业人员 (万人)'].pct_change() * 100 / df['GDP增长率 (%)']
        )['就业弹性'],
        'formula': '就业弹性 = 就业增长率 / GDP增长率'
    }
}
