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
    # ---- Profitability ----
    'Gross Margin': {
        'sub-category': 'profitability',
        'data': ['Revenue', 'CostOfGoodsAndServicesSold'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            GrossMargin=lambda df: (df['Revenue'] - df['CostOfGoodsAndServicesSold']) / df['Revenue'] * 100,
        )['GrossMargin'],
        'formula': '(Revenue - COGS) / Revenue * 100%',
        'note': 'Unit: %',
    },
    # 'Operating Margin': {
    #     'data': ['Revenue', 'OperatingIncomeLoss'],
    #     'function': lambda df: df.assign(
    #         CHECK=lambda df: check_condition(df['Revenue'] > 0),
    #         OperatingMargin=lambda df: df['OperatingIncomeLoss'] / df['Revenue'] * 100,
    #     )['OperatingMargin'],
    #     'formula': 'Operating Income / Revenue * 100%',
    #     'note': 'Unit: %',
    # },
    # 'EBIT Margin': {
    #     'data': ['Revenue', 'OperatingIncomeLoss'],
    #     'function': lambda df: df.assign(
    #         CHECK=lambda df: check_condition(df['Revenue'] > 0),
    #         EBITMargin=lambda df: df['OperatingIncomeLoss'] / df['Revenue'] * 100,
    #     )['EBITMargin'],
    #     'note': 'Unit: %; Use Operating Income as EBIT',
    #     'formula': 'EBIT / Revenue * 100% = Operating Income / Revenue * 100%'
    # },
    # 'EBITDA Margin': {
    #     'data': ['Revenue', 'OperatingIncomeLoss', 'DepreciationExpense', 'AmortizationOfIntangibles'],
    #     'function': lambda df: df.assign(
    #         CHECK=lambda df: check_condition(df['Revenue'] > 0),
    #         EBITDAMargin=lambda df: (df['OperatingIncomeLoss'] + df['DepreciationExpense'] + df['AmortizationOfIntangibles']) / df['Revenue'] * 100,
    #     )['EBITDAMargin'],
    #     'note': 'Unit: %; Use Operating Income as EBIT',
    #     'formula': '(Operating Income + D&A) / Revenue * 100%'
    # },
    'EBIT Margin': {
        'sub-category': 'profitability',
        'data': ['Revenue', 'PretaxIncomeLoss', 'InterestExpense'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            EBITMargin=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) / df['Revenue'] * 100,
        )['EBITMargin'],
        'note': 'Unit: %; Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': 'EBIT / Revenue * 100% = (Pretax Income + Interest Expense) / Revenue * 100%'
    },
    'EBITDA Margin': {
        'sub-category': 'profitability',
        'data': ['Revenue', 'PretaxIncomeLoss', 'InterestExpense', 'DepreciationExpense', 'AmortizationOfIntangibles'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            EBITDAMargin=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense'] + df['DepreciationExpense'] + df['AmortizationOfIntangibles']) / df['Revenue'] * 100,
        )['EBITDAMargin'],
        'note': 'Unit: %;  Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': '(Pretax Income + Interest Expense + D&A) / Revenue * 100%'
    },
    'Net Profit Margin': {
        'sub-category': 'profitability',
        'data': ['Revenue', 'NetIncome'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            NetProfitMargin=lambda df: df['NetIncome'] / df['Revenue'] * 100,
        )['NetProfitMargin'],
        'formula': 'Net Income / Revenue * 100%',
        'note': 'Unit: %'
    },
    'Return on Equity': {
        'sub-category': 'profitability',
        'data': ['NetIncome', 'AllEquityBalance'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['AllEquityBalance'] > 0),
            ROE=lambda df: df['NetIncome'] / ((df['AllEquityBalance'].shift(1) + df['AllEquityBalance']) / 2) * 100,
        )['ROE'],
        'note': 'Unit: %; Use the average of beginning and ending stockholders’ equity',
        'formula': 'Net Income / Average Stockholders Equity * 100%'
    },
    'Return on Assets': {
        'sub-category': 'profitability',
        'data': ['NetIncome', 'Assets'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Assets'] > 0),
            ROA=lambda df: df['NetIncome'] / ((df['Assets'].shift(1) + df['Assets']) / 2) * 100,
        )['ROA'],
        'note': 'Unit: %; Use the average of beginning and ending total assets',
        'formula': 'Net Income / Average Total Assets * 100%'
    },
    'Return on Invested Capital': {
        'sub-category': 'profitability',
        'data': ['PretaxIncomeLoss', 'InterestExpense', 'IncomeTaxes', 'AllEquityBalanceIncludingMinorityInterest', 'ShortTermDebt', 'LongTermDebt', 'OperatingLeaseCurrentDebtEquivalent', 'OperatingLeaseNonCurrentDebtEquivalent', 'CashAndMarketableSecurities', 'ShortTermInvestments'],
        'function': lambda df: df.assign(
            CHECK1=lambda df: check_condition((df['IncomeTaxes'] >= 0) & (df['PretaxIncomeLoss'] > 0)),
            NOPAT=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) * (1 - df['IncomeTaxes'] / df['PretaxIncomeLoss']),
            InvestedCapital=lambda df: df['AllEquityBalanceIncludingMinorityInterest'] + df['ShortTermDebt'] + df['LongTermDebt'] + df['OperatingLeaseCurrentDebtEquivalent'] + df['OperatingLeaseNonCurrentDebtEquivalent'] - df['CashAndMarketableSecurities'] - df['ShortTermInvestments'],
            CHECK2=lambda df: check_condition((df['InvestedCapital'].shift(1) + df['InvestedCapital']) != 0),
            ROIC=lambda df: df['NOPAT'] / ((df['InvestedCapital'].shift(1) + df['InvestedCapital']) / 2) * 100
        )['ROIC'],
        'note': 'Unit: %;  Use the sum of Pretax Income and Interest Expense as EBIT; Use the average of beginning and ending invested capital. Total cash is deducted rather than excess cash. Operating lease liabilities are capitalized and included in debt per ASC 842.',
        'formula': 'ROIC = NOPAT / Average Invested Capital * 100%; NOPAT = EBIT x (1 - effective tax rate); EBIT = Pretax Income + Interest Expense; effective tax rate = Income Taxes / Pretax Income; Invested Capital = Equity (incl. minority) + Total Debt (short + long-term debt + operating lease liabilities) - Cash and Short-Term Investments'
    },
    # ---- Efficiency / Activity ----
    'Days Inventory Outstanding': {
        'sub-category': 'operational efficiency',
        'data': ['CostOfGoodsAndServicesSold', 'Inventories'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['CostOfGoodsAndServicesSold'] > 0),
            DIO=lambda df: (df['Inventories'].shift(1) + df['Inventories']) / 2 / df['CostOfGoodsAndServicesSold'] * 365,
        )['DIO'],
        'note': 'Use the average of beginning and ending inventory; one year is assumed to have 365 days',
        'formula': 'Average Inventory / COGS * 365'
    },
    'Days Sales Outstanding': {
        'sub-category': 'operational efficiency',
        'data': ['Revenue', 'TradeReceivables'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            DSO=lambda df: (df['TradeReceivables'].shift(1) + df['TradeReceivables']) / 2 / df['Revenue'] * 365,
        )['DSO'],
        'note': 'Use the average of beginning and ending trade receivables; one year is assumed to have 365 days',
        'formula': 'Average Trade Receivables / Revenue * 365'
    },
    'Days Payable Outstanding': {
        'sub-category': 'operational efficiency',
        'data': ['CostOfGoodsAndServicesSold', 'TradePayables'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['CostOfGoodsAndServicesSold'] > 0),
            DPO=lambda df: (df['TradePayables'].shift(1) + df['TradePayables']) / 2 / df['CostOfGoodsAndServicesSold'] * 365,
        )['DPO'],
        'note': 'Use the average of beginning and ending trade payables; one year is assumed to have 365 days',
        'formula': 'Average Trade Payables / COGS * 365'
    },
    'Operating Cycle': {
        'sub-category': 'operational efficiency',
        'data': ['CostOfGoodsAndServicesSold', 'Inventories', 'Revenue', 'TradeReceivables'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['CostOfGoodsAndServicesSold'] > 0) & (df['Revenue'] > 0)),
            OperatingCycle=lambda df: (df['Inventories'].shift(1) + df['Inventories']) / 2 / df['CostOfGoodsAndServicesSold'] * 365 + (df['TradeReceivables'].shift(1) + df['TradeReceivables']) / 2 / df['Revenue'] * 365,
        )['OperatingCycle'],
        'note': 'Use the average of beginning and ending inventory and trade receivables; one year is assumed to have 365 days',
        'formula': 'Days Inventory Outstanding + Days Sales Outstanding'
    },
    'Cash Conversion Cycle': {
        'sub-category': 'operational efficiency',
        'data': ['CostOfGoodsAndServicesSold', 'Inventories', 'Revenue', 'TradeReceivables', 'TradePayables'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['CostOfGoodsAndServicesSold'] > 0) & (df['Revenue'] > 0)),
            CCC=lambda df: (df['Inventories'].shift(1) + df['Inventories']) / 2 / df['CostOfGoodsAndServicesSold'] * 365 + (df['TradeReceivables'].shift(1) + df['TradeReceivables']) / 2 / df['Revenue'] * 365 - (df['TradePayables'].shift(1) + df['TradePayables']) / 2 / df['CostOfGoodsAndServicesSold'] * 365,
        )['CCC'],
        'note': 'Use the average of beginning and ending inventory, trade receivables, and trade payables; one year is assumed to have 365 days',
        'formula': 'Days Inventory Outstanding + Days Sales Outstanding - Days Payable Outstanding'
    },
    'Inventory Turnover': {
        'sub-category': 'operational efficiency',
        'data': ['CostOfGoodsAndServicesSold', 'Inventories'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Inventories'] > 0),
            InventoryTurnover=lambda df: df['CostOfGoodsAndServicesSold'] / ((df['Inventories'].shift(1) + df['Inventories']) / 2),
        )['InventoryTurnover'],
        'note': 'Use the average of beginning and ending inventory',
        'formula': 'COGS / Average Inventory'
    },
    'Receivables Turnover': {
        'sub-category': 'operational efficiency',
        'data': ['Revenue', 'TradeReceivables'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['TradeReceivables'] > 0),
            ReceivablesTurnover=lambda df: df['Revenue'] / ((df['TradeReceivables'].shift(1) + df['TradeReceivables']) / 2),
        )['ReceivablesTurnover'],
        'note': 'Use the average of beginning and ending trade receivables',
        'formula': 'Revenue / Average Trade Receivables'
    },
    'Payables Turnover': {
        'sub-category': 'operational efficiency',
        'data': ['CostOfGoodsAndServicesSold', 'TradePayables'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['TradePayables'] > 0),
            PayablesTurnover=lambda df: df['CostOfGoodsAndServicesSold'] / ((df['TradePayables'].shift(1) + df['TradePayables']) / 2),
        )['PayablesTurnover'],
        'note': 'Use the average of beginning and ending trade payables',
        'formula': 'COGS / Average Trade Payables'
    },
    'Total Asset Turnover': {
        'sub-category': 'operational efficiency',
        'data': ['Revenue', 'Assets'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Assets'] > 0),
            TotalAssetTurnover=lambda df: df['Revenue'] / ((df['Assets'].shift(1) + df['Assets']) / 2),
        )['TotalAssetTurnover'],
        'note': 'Use the average of beginning and ending total assets',
        'formula': 'Revenue / Average Total Assets'
    },
    'Fixed Asset Turnover': {
        'sub-category': 'operational efficiency',
        'data': ['Revenue', 'PlantPropertyEquipmentNet'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['PlantPropertyEquipmentNet'] > 0),
            FixedAssetTurnover=lambda df: df['Revenue'] / ((df['PlantPropertyEquipmentNet'].shift(1) + df['PlantPropertyEquipmentNet']) / 2),
        )['FixedAssetTurnover'],
        'note': 'Use the average of beginning and ending fixed assets',
        'formula': 'Revenue / Average Net PP&E'
    },
    'Working Capital Turnover': {
        'sub-category': 'operational efficiency',
        'data': ['Revenue', 'CurrentAssetsTotal', 'CurrentLiabilitiesTotal'],
        'function': lambda df: df.assign(
            NWC=lambda df: df['CurrentAssetsTotal'] - df['CurrentLiabilitiesTotal'],
            AvgNWC=lambda df: (df['NWC'].shift(1) + df['NWC']) / 2,
            CHECK=lambda df: check_condition(df['AvgNWC'] != 0),
            WorkingCapitalTurnover=lambda df: df['Revenue'] / df['AvgNWC'],
        )['WorkingCapitalTurnover'],
        'note': 'Use the average of beginning and ending net working capital',
        'formula': 'Revenue / Average Net Working Capital'
    },
    # ---- Solvency / Leverage ----
    'Current Ratio': {
        'sub-category': 'solvency',
        'data': ['CurrentAssetsTotal', 'CurrentLiabilitiesTotal'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['CurrentLiabilitiesTotal'] > 0),
            CurrentRatio=lambda df: df['CurrentAssetsTotal'] / df['CurrentLiabilitiesTotal'],
        )['CurrentRatio'],
        'formula': 'Current Assets / Current Liabilities'
    },
    'Quick Ratio': {
        'sub-category': 'solvency',
        'data': ['CurrentAssetsTotal', 'CurrentLiabilitiesTotal', 'Inventories'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['CurrentLiabilitiesTotal'] > 0),
            QuickRatio=lambda df: (df['CurrentAssetsTotal'] - df['Inventories']) / df['CurrentLiabilitiesTotal'],
        )['QuickRatio'],
        'formula': '(Current Assets - Inventory) / Current Liabilities'
    },
    'Cash Ratio': {
        'sub-category': 'solvency',
        'data': ['CashAndMarketableSecurities', 'ShortTermInvestments', 'CurrentLiabilitiesTotal'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['CurrentLiabilitiesTotal'] > 0),
            CashRatio=lambda df: (df['CashAndMarketableSecurities'] + df['ShortTermInvestments']) / df['CurrentLiabilitiesTotal'],
        )['CashRatio'],
        'formula': '(Cash + Short-Term Investments) / Current Liabilities'
    },
    'Liability-to-Asset Ratio': {
        'sub-category': 'solvency',
        'data': ['Liabilities', 'Assets'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Assets'] > 0),
            LiabilityToAsset=lambda df: df['Liabilities'] / df['Assets'] * 100,
        )['LiabilityToAsset'],
        'formula': 'Total Liabilities / Total Assets * 100%',
        'note': 'Unit: %'
    },
    'Liability-to-Equity Ratio': {
        'sub-category': 'solvency',
        'data': ['Liabilities', 'AllEquityBalanceIncludingMinorityInterest'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['AllEquityBalanceIncludingMinorityInterest'] > 0),
            LiabilityToEquity=lambda df: df['Liabilities'] / df['AllEquityBalanceIncludingMinorityInterest'] * 100,
        )['LiabilityToEquity'],
        'formula': 'Total Liabilities / Total Equity * 100%',
        'note': 'Unit: %; Total equity includes minority interest'
    },
    'Net Debt to Equity': {
        'sub-category': 'solvency',
        'data': ['ShortTermDebt', 'LongTermDebt', 'OperatingLeaseCurrentDebtEquivalent', 'OperatingLeaseNonCurrentDebtEquivalent', 'CashAndMarketableSecurities', 'ShortTermInvestments', 'AllEquityBalanceIncludingMinorityInterest'],
        'function': lambda df: df.assign(
            NetDebt=lambda df: df['ShortTermDebt'] + df['LongTermDebt'] + df['OperatingLeaseCurrentDebtEquivalent'] + df['OperatingLeaseNonCurrentDebtEquivalent'] - df['CashAndMarketableSecurities'] - df['ShortTermInvestments'],
            CHECK=lambda df: check_condition(df['AllEquityBalanceIncludingMinorityInterest'] > 0),
            NetDebtToEquity=lambda df: df['NetDebt'] / df['AllEquityBalanceIncludingMinorityInterest']
        )['NetDebtToEquity'],
        'formula': 'Net Debt / Total Equity; Net Debt = Total Debt (short + long-term debt + operating lease liabilities) - Cash - Short-Term Investments'
    },
    'Interest Coverage Ratio': {
        'sub-category': 'solvency',
        'data': ['PretaxIncomeLoss', 'InterestExpense'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['InterestExpense'] > 0),
            InterestCoverage=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) / df['InterestExpense'],
        )['InterestCoverage'],
        'note': ' Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': 'EBIT / Interest Expense = (Pretax Income + Interest Expense) / Interest Expense'
    },
    'Net Debt to EBITDA': {
        'sub-category': 'solvency',
        'data': ['ShortTermDebt', 'LongTermDebt', 'OperatingLeaseCurrentDebtEquivalent', 'OperatingLeaseNonCurrentDebtEquivalent', 'CashAndMarketableSecurities', 'ShortTermInvestments', 'PretaxIncomeLoss', 'InterestExpense', 'DepreciationExpense', 'AmortizationOfIntangibles'],
        'function': lambda df: df.assign(
            NetDebt=lambda df: df['ShortTermDebt'] + df['LongTermDebt'] + df['OperatingLeaseCurrentDebtEquivalent'] + df['OperatingLeaseNonCurrentDebtEquivalent'] - df['CashAndMarketableSecurities'] - df['ShortTermInvestments'],
            EBITDA=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) + df['DepreciationExpense'] + df['AmortizationOfIntangibles'],
            CHECK=lambda df: check_condition(df['EBITDA'] > 0),
            NetDebtToEBITDA=lambda df: df['NetDebt'] / df['EBITDA']
        )['NetDebtToEBITDA'],
        'note': ' Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': 'Net Debt / EBITDA; Net Debt = Total Debt (short + long-term debt + operating lease liabilities) - Cash - Short-Term Investments; EBITDA = Pretax Income + Interest Expense + D&A'
    },
    # ---- Growth ----
    'Revenue Growth': {
        'sub-category': 'growth',
        'data': ['Revenue'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            RevenueGrowth=lambda df: df['Revenue'].diff() / df['Revenue'].shift(1) * 100,
        )['RevenueGrowth'],
        'formula': '(Revenue_t - Revenue_{t-1}) / Revenue_{t-1} * 100%',
        'note': 'Unit: %'
    },
    'EBITDA Growth': {
        'sub-category': 'growth',
        'data': ['PretaxIncomeLoss', 'InterestExpense', 'DepreciationExpense', 'AmortizationOfIntangibles'],
        'function': lambda df: df.assign(
            EBITDA=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) + df['DepreciationExpense'] + df['AmortizationOfIntangibles'],
            CHECK=lambda df: check_condition(df['EBITDA'] > 0),
            EBITDAGrowth=lambda df: df['EBITDA'].diff() / df['EBITDA'].shift(1) * 100
        )['EBITDAGrowth'],
        'note': 'Unit: %; Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': '(EBITDA_t - EBITDA_{t-1}) / EBITDA_{t-1} * 100%; EBITDA = Pretax Income + Interest Expense + D&A'
    },
    'Net Income Growth': {
        'sub-category': 'growth',
        'data': ['NetIncome'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['NetIncome'].shift(1) != 0),
            NetIncomeGrowth=lambda df: df['NetIncome'].diff() / df['NetIncome'].shift(1).abs() * 100,
        )['NetIncomeGrowth'],
        'formula': '(Net Income_t - Net Income_{t-1}) / |Net Income_{t-1}| * 100%',
        'note': 'Unit: %'
    },
    'EPS Growth': {
        'sub-category': 'growth',
        'data': ['NetIncome', 'SharesAverage'],
        'function': lambda df: df.assign(
            EPS=lambda df: df['NetIncome'] / df['SharesAverage'],
            CHECK=lambda df: check_condition(df['EPS'].shift(1) != 0),
            EPSGrowth=lambda df: df['EPS'].diff() / df['EPS'].shift(1).abs() * 100
        )['EPSGrowth'],
        'formula': '(EPS_t - EPS_{t-1}) / |EPS_{t-1}| * 100%; EPS = Net Income / Weighted-Average Basic Shares',
        'note': 'Unit: %'
    },
    # ---- Cash Flow ----
    'Free Cash Flow': {
        'sub-category': 'cash flow',
        'data': ['NetCashFromOperatingActivities', 'CapitalExpenses'],
        'function': lambda df: df['NetCashFromOperatingActivities'] - df['CapitalExpenses'],
        'formula': 'CFO - Capex',
        'note': 'Unit: $'
    },
    'Free Cash Flow to Firm': {
        'sub-category': 'cash flow',
        'data': ['NetCashFromOperatingActivities', 'CapitalExpenses', 'InterestExpense', 'IncomeTaxes', 'PretaxIncomeLoss'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['IncomeTaxes'] >= 0) & (df['PretaxIncomeLoss'] > 0)),
            FCFF=lambda df: df['NetCashFromOperatingActivities'] - df['CapitalExpenses'] + df['InterestExpense'] * (1 - df['IncomeTaxes'] / df['PretaxIncomeLoss']),
        )['FCFF'],
        'formula': 'CFO - Capex + Interest Expense x (1 - effective tax rate)',
        'note': 'Unit: $'
    },
    'Free Cash Flow to Equity': {
        'sub-category': 'cash flow',
        'data': ['NetCashFromOperatingActivities', 'CapitalExpenses', 'ShortTermDebt', 'LongTermDebt'],
        'function': lambda df: df.assign(
            TotalDebt=lambda df: df['ShortTermDebt'] + df['LongTermDebt'],
            NetBorrowing=lambda df: df['TotalDebt'].diff(),
            FCFE=lambda df: df['NetCashFromOperatingActivities'] - df['CapitalExpenses'] + df['NetBorrowing']
        )['FCFE'],
        'note': 'Unit: $; Net Borrowing is approximated as the year-over-year change in (short-term debt + long-term debt)',
        'formula': 'CFO - Capex + Net Borrowing; Net Borrowing = delta(Short-Term Debt + Long-Term Debt)'
    },
    'Operating Cash Flow Margin': {
        'sub-category': 'cash flow',
        'data': ['NetCashFromOperatingActivities', 'Revenue'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            OCFMargin=lambda df: df['NetCashFromOperatingActivities'] / df['Revenue'] * 100,
        )['OCFMargin'],
        'formula': 'CFO / Revenue * 100%',
        'note': 'Unit: %'
    },
    'Capex Intensity': {
        'sub-category': 'cash flow',
        'data': ['CapitalExpenses', 'Revenue'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['Revenue'] > 0),
            CapexIntensity=lambda df: df['CapitalExpenses'] / df['Revenue'] * 100,
        )['CapexIntensity'],
        'formula': 'Capex / Revenue * 100%',
        'note': 'Unit: %'
    },
    'Cash Flow to Net Income': {
        'sub-category': 'cash flow',
        'data': ['NetCashFromOperatingActivities', 'NetIncome'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['NetIncome'] != 0),
            CFToNetIncome=lambda df: df['NetCashFromOperatingActivities'] / df['NetIncome'],
        )['CFToNetIncome'],
        'formula': 'CFO / Net Income'
    },
    'Cash Flow to Debt': {
        'sub-category': 'cash flow',
        'data': ['NetCashFromOperatingActivities', 'ShortTermDebt', 'LongTermDebt'],
        'function': lambda df: df.assign(
            TotalDebt=lambda df: df['ShortTermDebt'] + df['LongTermDebt'],
            CHECK=lambda df: check_condition(df['TotalDebt'] != 0),
            CFToDebt=lambda df: df['NetCashFromOperatingActivities'] / df['TotalDebt'],
        )['CFToDebt'],
        'formula': 'CFO / Total Debt; Total Debt = Short-Term Debt + Long-Term Debt'
    },
    # ---- Per Share ----
    'Earnings Per Share Basic': {
        'sub-category': 'per share',
        'data': ['NetIncome', 'SharesAverage', 'PreferredDividendExpense'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['SharesAverage'] > 0),
            EPSBasic=lambda df: (df['NetIncome'] - df['PreferredDividendExpense']) / df['SharesAverage'],
        )['EPSBasic'],
        'formula': '(Net Income - Preferred Dividends) / Weighted-Average Basic Shares'
    },
    'Earnings Per Share Diluted': {
        'sub-category': 'per share',
        'data': ['NetIncome', 'SharesFullyDilutedAverage', 'PreferredDividendExpense'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['SharesFullyDilutedAverage'] > 0),
            EPSDiluted=lambda df: (df['NetIncome'] - df['PreferredDividendExpense']) / df['SharesFullyDilutedAverage'],
        )['EPSDiluted'],
        'formula': '(Net Income - Preferred Dividends) / Weighted-Average Diluted Shares'
    },
    'Operating Cash Flow Per Share': {
        'sub-category': 'per share',
        'data': ['NetCashFromOperatingActivities', 'SharesAverage'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['SharesAverage'] > 0),
            OCFPerShare=lambda df: df['NetCashFromOperatingActivities'] / df['SharesAverage'],
        )['OCFPerShare'],
        'formula': 'CFO / Weighted-Average Basic Shares'
    },
    # ---- DuPont decomposition ----
    # Return on Equity = Net Profit Margin x Total Asset Turnover x Equity Multiplier
    'Equity Multiplier': {
        'sub-category': 'solvency',
        'data': ['Assets', 'AllEquityBalance'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition(df['AllEquityBalance'] > 0),
            EquityMultiplier=lambda df: (df['Assets'].shift(1) + df['Assets']) / (df['AllEquityBalance'].shift(1) + df['AllEquityBalance']),
        )['EquityMultiplier'],
        'note': 'Use the average of beginning and ending values',
        'formula': 'Average Total Assets / Average Stockholders Equity'
    },
    # ---- Valuation building blocks ----
    'EBITDA': {
        'sub-category': 'valuation',
        'data': ['PretaxIncomeLoss', 'InterestExpense', 'DepreciationExpense', 'AmortizationOfIntangibles'],
        'function': lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) + df['DepreciationExpense'] + df['AmortizationOfIntangibles'],
        'note': 'Unit: $; Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': 'EBITDA = EBIT + D&A = Pretax Income + Interest Expense + D&A'
    },
    'NOPAT': {
        'sub-category': 'valuation',
        'data': ['PretaxIncomeLoss', 'InterestExpense', 'IncomeTaxes'],
        'function': lambda df: df.assign(
            CHECK=lambda df: check_condition((df['IncomeTaxes'] >= 0) & (df['PretaxIncomeLoss'] > 0)),
            NOPAT=lambda df: (df['PretaxIncomeLoss'] + df['InterestExpense']) * (1 - df['IncomeTaxes'] / df['PretaxIncomeLoss']),
        )['NOPAT'],
        'note': 'Unit: $; Use the sum of Pretax Income and Interest Expense as EBIT',
        'formula': 'NOPAT = EBIT x (1 - effective tax rate); EBIT = Pretax Income + Interest Expense; effective tax rate = Income Taxes / Pretax Income'
    }
}



def _calc_KAMA(df, n_er=10, n_fast=2, n_slow=30):
    close = df['close'].values
    length = len(close)

    er = np.full(length, np.nan)
    sc = np.full(length, np.nan)
    kama = np.full(length, np.nan)

    # Constants
    fast_sc = 2 / (n_fast + 1)
    slow_sc = 2 / (n_slow + 1)

    # 1. Direction: net change over n_er periods
    direction = np.abs(close[n_er:] - close[:-n_er])  # length = length - n_er

    # 2. Volatility: sum of absolute changes over n_er periods
    volatility = np.abs(np.diff(close))
    vol_sum = np.convolve(volatility, np.ones(n_er), mode='valid')  # length = length - n_er

    with np.errstate(divide='ignore', invalid='ignore'):
        er_values = np.where(vol_sum != 0, direction / vol_sum, 0.0)

    er[n_er:] = er_values

    # 3. Smoothing Constant
    sc[n_er:] = (er[n_er:] * (fast_sc - slow_sc) + slow_sc) ** 2

    # 4. Seed value
    kama[n_er - 1] = np.mean(close[:n_er])

    # 5. Recursive update
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

    # Initialize state
    trend = 1 if close[1] > close[0] else -1
    ep = high[1] if trend == 1 else low[1]
    af = initial_af
    sar[1] = low[0] if trend == 1 else high[0]

    for i in range(2, length):
        # 1. Compute the theoretical next SAR value
        next_sar = sar[i-1] + af * (ep - sar[i-1])

        # 2. Handle reversal or state update depending on the current trend
        if trend == 1:
            if low[i] < next_sar:
                # Reversal to downtrend
                trend = -1
                next_sar = ep  # The prior high becomes the new SAR
                ep = low[i]    # Reset extreme point to the current low
                af = initial_af
            else:
                # Uptrend continues: update extreme point and acceleration factor
                if high[i] > ep:
                    ep = high[i]
                    af = min(max_af, af + step_af)

            # 3. Boundary control: SAR cannot penetrate the prior two periods' lows
            sar[i] = min(next_sar, low[i-1], low[i-2])

        else: # trend == -1
            if high[i] > next_sar:
                # Reversal to uptrend
                trend = 1
                next_sar = ep  # The prior low becomes the new SAR
                ep = high[i]   # Reset extreme point to the current high
                af = initial_af
            else:
                # Downtrend continues: update extreme point and acceleration factor
                if low[i] < ep:
                    ep = low[i]
                    af = min(max_af, af + step_af)

            # 3. Boundary control: SAR cannot penetrate the prior two periods' highs
            sar[i] = max(next_sar, high[i-1], high[i-2])

    return pd.Series(sar, index=df.index)

def _calc_supertrend(df, n=10, m=3):
    '''
    Standard SuperTrend calculation using Wilder's RMA smoothing of ATR.
    Outputs Middle, Upper, and Lower bands.
    '''
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    length = len(c)

    # Initialize output arrays
    middle = (h + l) / 2.0
    final_upper = np.full(length, np.nan)
    final_lower = np.full(length, np.nan)

    if length < n:
        return pd.DataFrame({'Middle': middle, 'Upper': final_upper, 'Lower': final_lower}, index=df.index)

    # 1. True Range
    tr = np.zeros(length)
    tr[0] = h[0] - l[0]
    for i in range(1, length):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    # 2. ATR using Wilder smoothing (RMA)
    atr = np.zeros(length)
    atr[n-1] = np.mean(tr[:n])  # Seed value: simple average of the first n bars
    for i in range(n, length):
        atr[i] = (atr[i-1] * (n - 1) + tr[i]) / n

    # 3. Basic bands
    basic_upper = middle + m * atr
    basic_lower = middle - m * atr

    # 4. State machine (core logic)
    # Initial state assumes long (1 = long, -1 = short)
    trend = 1
    curr_upper = basic_upper[n-1]
    curr_lower = basic_lower[n-1]

    for i in range(n, length):
        if trend == 1:
            # Long state: lower band can only rise, never fall
            curr_lower = max(basic_lower[i], curr_lower)

            # Reversal trigger: close breaks below the active lower band
            if c[i] < curr_lower:
                trend = -1
                curr_upper = basic_upper[i]  # Reset upper band at the moment of reversal
                # Note: after reversal curr_lower stays frozen as overhead resistance
            # else: long continues, curr_upper stays frozen

        else: # trend == -1
            # Short state: upper band can only fall, never rise
            curr_upper = min(basic_upper[i], curr_upper)

            # Reversal trigger: close breaks above the active upper band
            if c[i] > curr_upper:
                trend = 1
                curr_lower = basic_lower[i]  # Reset lower band at the moment of reversal
                # Note: after reversal curr_upper stays frozen as underlying support
            # else: short continues, curr_lower stays frozen

        # Record final band values
        final_upper[i] = curr_upper
        final_lower[i] = curr_lower

    return pd.DataFrame({
        'Middle': middle,
        'Upper': final_upper,
        'Lower': final_lower
    }, index=df.index)

indicators_technical_ma = {
    'Simple Moving Average': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df['close'].rolling(window=n).mean(),
        'formula': r'SMA_n = 1 / n * \sum_{i=0}^{n-1} P_{t-i}'
    },
    'Weighted Moving Average': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df['close'].rolling(window=n).apply(lambda x: (x * range(1, n + 1)).sum() / sum(range(1, n + 1)), raw=True),
        'note': 'Weights are 1, 2, ..., n',
        'formula': r'WMA_n = \sum_{i=0}^{n-1} (n-i) * P_{t-i} / \sum_{j=1}^{n} j'
    },
    'Exponential Moving Average': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df['close'].ewm(span=n, min_periods=n, adjust=False).mean(),
        'note': 'Smoothing factor alpha = 2 / (n + 1); Require 4n data points to warm up; Use the first data point as the initial value',
        'data_lookback_n': lambda n: n * 4,
        'formula': r'EMA_t = \alpha * P_t + (1 - \alpha) * EMA_{t-1}, \quad \alpha = 2 / (n + 1)'
    },
    'Double Exponential Moving Average': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df.assign(
            EMA1=lambda df: df['close'].ewm(span=n, min_periods=n, adjust=False).mean(),
            EMA2=lambda df: df['EMA1'].ewm(span=n, min_periods=n, adjust=False).mean(),
            DEMA=lambda df: 2 * df['EMA1'] - df['EMA2']
        )['DEMA'],
        'note': 'Smoothing factor alpha = 2 / (n + 1); Require 5n data points to warm up; Use the first data point as the initial value',
        'data_lookback_n': lambda n: n * 5,
        'formula': 'DEMA = 2 * EMA1 - EMA2; EMA1 = EMA_n(P); EMA2 = EMA_n(EMA1)'
    },
    'Triple Exponential Moving Average': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': lambda df, n: df.assign(
            EMA1=lambda df: df['close'].ewm(span=n, min_periods=n, adjust=False).mean(),
            EMA2=lambda df: df['EMA1'].ewm(span=n, min_periods=n, adjust=False).mean(),
            EMA3=lambda df: df['EMA2'].ewm(span=n, min_periods=n, adjust=False).mean(),
            TEMA=lambda df: 3 * df['EMA1'] - 3 * df['EMA2'] + df['EMA3']
        )['TEMA'],
        'note': 'Smoothing factor alpha = 2 / (n + 1); Require 6n data points to warm up; Use the first data point as the initial value',
        'data_lookback_n': lambda n: n * 6,
        'formula': 'TEMA = 3 * EMA1 - 3 * EMA2 + EMA3; EMA1 = EMA_n(P); EMA2 = EMA_n(EMA1); EMA3 = EMA_n(EMA2)'
    },
    'Hull Moving Average': {
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
    'Bollinger Bands': {
        'sub-category': 'channel',
        'data': ['close'],
        'function': lambda df, n=20: df.assign(
            Middle=lambda df: df['close'].rolling(window=n).mean(),
            STD=lambda df: df['close'].rolling(window=n).std(ddof=0),
            Upper=lambda df: df['Middle'] + 2 * df['STD'],
            Lower=lambda df: df['Middle'] - 2 * df['STD']
        )[['Upper', 'Middle', 'Lower']],
        'note': 'Standard parameters: middle band SMA(20), band width multiplier 2, use population standard deviation',
        'formula': r'Middle = SMA_n(P); Upper = Middle + 2 * \sigma_n(P); Lower = Middle - 2 * \sigma_n(P)'
    },
    'Keltner Channel': {
        'sub-category': 'channel',
        'data': ['close', 'high', 'low'],
        'function': lambda df, n_ema=20, n_atr=10, multiplier=1.5: df.assign(
            Middle=lambda df: df['close'].ewm(span=n_ema, min_periods=n_ema, adjust=False).mean(),
            ATR=lambda df: pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1).ewm(alpha=1/n_atr, min_periods=n_atr, adjust=False).mean(),
            Upper=lambda df: df['Middle'] + multiplier * df['ATR'],
            Lower=lambda df: df['Middle'] - multiplier * df['ATR']
        )[['Upper', 'Middle', 'Lower']],
        'note': 'Standard parameters: middle EMA(20), requiring 4n data points to warm up, using the first data point as the initial value; ATR(10) using Wilder smoothing, multiplier 1.5',
        'data_lookback_n': lambda n_ema=20, **kwargs: n_ema * 4,
        'formula': 'Middle = EMA_{n_{ema}}(C); Upper = Middle + 1.5 * ATR_{n_{atr}}^{Wilder}; Lower = Middle - 1.5 * ATR_{n_{atr}}^{Wilder}'
    },
    'Moving Average Envelope': {
        'sub-category': 'channel',
        'data': ['close'],
        'function': lambda df, n=20: df.assign(
            Middle=lambda df: df['close'].rolling(window=n).mean(),
            Upper=lambda df: df['Middle'] * 1.1,
            Lower=lambda df: df['Middle'] * 0.9
        )[['Upper', 'Middle', 'Lower']],
        'note': 'Standard parameters: middle band SMA(20), upper and lower bands +/- 10%',
        'formula': 'Middle = SMA_n(P); Upper = Middle * 1.1; Lower = Middle * 0.9'
    },
    'Donchian Channel': {
        'sub-category': 'channel',
        'data': ['high', 'low'],
        'function': lambda df, n=20: df.assign(
            Upper=lambda df: df['high'].rolling(window=n).max(),
            Lower=lambda df: df['low'].rolling(window=n).min(),
            Middle=lambda df: (df['Upper'] + df['Lower']) / 2
        )[['Upper', 'Middle', 'Lower']],
        'note': 'Standard parameters: 20-period high/low, middle band is their average',
        'formula': 'Upper = max_{i=0}^{n-1} H_{t-i}; Lower = min_{i=0}^{n-1} L_{t-i}; Middle = (Upper + Lower) / 2'
    },
    'SuperTrend': {
        'sub-category': 'channel',
        'data': ['high', 'low', 'close'],
        'function': _calc_supertrend,
        'note': 'Standard parameters: period 10, multiplier 3; Require 5n data points to warm up',
        'data_lookback_n': lambda n=10, **kwargs: n * 5,
        'formula': 'Mid_t = (H_t + L_t) / 2; BasicUpper_t = Mid_t + m * ATR_n^{Wilder}; BasicLower_t = Mid_t - m * ATR_n^{Wilder}; in long state Lower_t = max(BasicLower_t, Lower_{t-1}), flip to short if C_t < Lower_t; in short state Upper_t = min(BasicUpper_t, Upper_{t-1}), flip to long if C_t > Upper_t'
    },
}

indicators_technical_n = {
    'Aroon Up': {
        'sub-category': 'trend',
        'data': ['high'],
        'function': lambda df, n=25: df.assign(
            AroonUp=lambda df: df['high'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmax()) / n * 100, raw=True),
        )['AroonUp'],
        'note': 'Standard period: 25',
        'data_lookback_n': lambda n=25: n,
        'formula': 'AroonUp = (n - i_{max}) / n * 100; i_{max} = argmax_{i=0}^{n} H_{t-i}, the number of bars since the highest high in the last n+1 bars'
    },
    'Aroon Down': {
        'sub-category': 'trend',
        'data': ['high', 'low'],
        'function': lambda df, n=25: df.assign(
            AroonDown=lambda df: df['low'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmin()) / n * 100, raw=True),
        )['AroonDown'],
        'note': 'Standard period: 25',
        'data_lookback_n': lambda n=25: n,
        'formula': 'AroonDown = (n - i_{min}) / n * 100; i_{min} = argmin_{i=0}^{n} L_{t-i}, the number of bars since the lowest low in the last n+1 bars'
    },
    'Aroon Oscillator': {
        'sub-category': 'trend',
        'data': ['high', 'low'],
        'function': lambda df, n=25: df.assign(
            AroonUp=lambda df: df['high'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmax()) / n * 100, raw=True),
            AroonDown=lambda df: df['low'].rolling(window=n+1).apply(lambda x: (n - x[::-1].argmin()) / n * 100, raw=True),
            AroonOsc=lambda df: df['AroonUp'] - df['AroonDown']
        )['AroonOsc'],
        'note': 'Standard period: 25',
        'data_lookback_n': lambda n=25: n,
        'formula': 'AroonOsc = AroonUp - AroonDown; AroonUp = (n - i_{max}) / n * 100; AroonDown = (n - i_{min}) / n * 100'
    },
    'Relative Strength Index': {
        'sub-category': 'momentum',
        'data': ['close'],
        'function': lambda df, n=14: df.assign(
            Change=lambda df: df['close'].diff(),
            Gain=lambda df: df['Change'].clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean(),
            Loss=lambda df: -df['Change'].clip(upper=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean(),
            RSI=lambda df: 100 - (100 / (1 + df['Gain'] / df['Loss']))
        )['RSI'],
        'note': 'Standard period: 14; Gain/Loss calculation uses Wilder Smoothing, using the first data point as the initial value; Require 4n data points to warm up',
        'data_lookback_n': lambda n=14: n * 4,
        'formula': 'RSI = 100 - (100 / (1 + Gain / Loss)); Gain = Wilder_n(max(Change, 0)); Loss = Wilder_n(-min(Change, 0)); Wilder Smoothing alpha = 1/n'
    },
    'Commodity Channel Index': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=20: df.assign(
            TP=lambda df: (df['high'] + df['low'] + df['close']) / 3,
            SMA_TP=lambda df: df['TP'].rolling(window=n).mean(),
            MeanDev=lambda df: df['TP'].rolling(window=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True),
            CCI=lambda df: (df['TP'] - df['SMA_TP']) / (0.015 * df['MeanDev'])
        )['CCI'],
        'note': 'Standard period: 20',
        'formula': r'TP_t = (H_t + L_t + C_t) / 3; MD_t = 1 / n * \sum_{i=0}^{n-1} |TP_{t-i} - SMA_n(TP)|; CCI = (TP_t - SMA_n(TP)) / (0.015 * MD_t)'
    },
    'Williams %R': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=14: df.assign(
            Highest=lambda df: df['high'].rolling(window=n).max(),
            Lowest=lambda df: df['low'].rolling(window=n).min(),
            WilliamsR=lambda df: (df['Highest'] - df['close']) / (df['Highest'] - df['Lowest']) * (-100)
        )['WilliamsR'],
        'note': 'Standard period: 14',
        'formula': r'%R = (max_{i=0}^{n-1} H_{t-i} - C_t) / (max_{i=0}^{n-1} H_{t-i} - min_{i=0}^{n-1} L_{t-i}) * (-100)'
    },
    'Bollinger %B': {
        'sub-category': 'volatility',
        'data': ['close'],
        'function': lambda df, n=20: df.assign(
            Middle=lambda df: df['close'].rolling(window=n).mean(),
            STD=lambda df: df['close'].rolling(window=n).std(ddof=0),
            Upper=lambda df: df['Middle'] + 2 * df['STD'],
            Lower=lambda df: df['Middle'] - 2 * df['STD'],
            PctB=lambda df: (df['close'] - df['Lower']) / (df['Upper'] - df['Lower'])
        )['PctB'],
        'note': 'Standard parameters: period 20, multiplier 2, use population standard deviation',
        'formula': r'Middle = SMA_n(P); Upper = Middle + 2 * \sigma_n(P); Lower = Middle - 2 * \sigma_n(P); %B = (P_t - Lower) / (Upper - Lower)'
    },
    'Average True Range': {
        'sub-category': 'volatility',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=14: pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1).ewm(alpha=1/n, min_periods=n, adjust=False).mean(),
        'note': 'Standard period: 14; ATR calculation uses Wilder Smoothing, using the first data point as the initial value; Require 4n data points to warm up',
        'data_lookback_n': lambda n=14: n * 4,
        'formula': 'TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|); ATR = Wilder_n(TR); Wilder Smoothing alpha = 1/n'
    },
    'Ease of Movement': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'volume'],
        'function': lambda df, n=14: df.assign(
            HL_Range=lambda df: np.where(df['high'] == df['low'], 1e-10, df['high'] - df['low']),
            MidMove=lambda df: ((df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2) / ((df['volume'] / 10**6) / df['HL_Range']),
            EMV=lambda df: df['MidMove'].rolling(window=n).mean()
        )['EMV'],
        'note': 'Standard period: 14; volume is scaled in millions in the denominator to avoid extreme values',
        'data_lookback_n': lambda n=14: n,
        'formula': 'MidMove_t = ((H_t + L_t) / 2 - (H_{t-1} + L_{t-1}) / 2) / ((V_t / 10^6) / (H_t - L_t)); EMV = SMA_n(MidMove)'
    },
    'Money Flow Index': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df, n=14: df.assign(
            TP=lambda df: (df['high'] + df['low'] + df['close']) / 3,
            MF=lambda df: df['TP'] * df['volume'],
            PosMF=lambda df: np.where(df['TP'] > df['TP'].shift(1), df['MF'], 0) * np.where(df['TP'].shift(1).isna(), np.nan, 1),  # first bar set to NaN
            NegMF=lambda df: np.where(df['TP'] < df['TP'].shift(1), df['MF'], 0) * np.where(df['TP'].shift(1).isna(), np.nan, 1),  # first bar set to NaN
            PosSum=lambda df: df['PosMF'].rolling(window=n).sum(),
            NegSum=lambda df: df['NegMF'].rolling(window=n).sum(),
            MFI=lambda df: 100 - (100 / (1 + df['PosSum'] / df['NegSum'].replace(0, np.nan)))
        )['MFI'],
        'note': 'Standard period: 14',
        'data_lookback_n': lambda n=14: n,
        'formula': r'TP_t = (H_t + L_t + C_t) / 3; MF_t = TP_t * V_t; PMF_t = MF_t * \mathbb{1}[TP_t > TP_{t-1}]; NMF_t = MF_t * \mathbb{1}[TP_t < TP_{t-1}]; MFR = \sum_{i=0}^{n-1} PMF_{t-i} / \sum_{i=0}^{n-1} NMF_{t-i}; MFI = 100 - 100 / (1 + MFR)'
    },
    'Stochastic %K': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=14, n_k=3: df.assign(
            RSV=lambda df: (df['close'] - df['low'].rolling(window=n).min()) / (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()) * 100,
            K=lambda df: df['RSV'].rolling(window=n_k).mean(),
        )['K'],
        'note': 'Standard parameters: 14, 3, 3; Calculates Slow Stochastic by default',
        'data_lookback_n': lambda n=14, n_k=3: n + n_k * 2,
        'formula': 'K = SMA(RSV, k); RSV = (C - Low_n) / (High_n - Low_n) * 100'
    },
    'Stochastic %D': {
        'sub-category': 'momentum',
        'data': ['high', 'low', 'close'],
        'function': lambda df, n=14, n_k=3, n_d=3: df.assign(
            RSV=lambda df: (df['close'] - df['low'].rolling(window=n).min()) / (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()) * 100,
            K=lambda df: df['RSV'].rolling(window=n_k).mean(),
            D=lambda df: df['K'].rolling(window=n_d).mean()
        )['D'],
        'note': 'Standard parameters: 14, 3, 3',
        'data_lookback_n': lambda n=14, n_k=3, n_d=3: n + n_k * 2 + n_d,
        'formula': 'D = SMA(K, k); K = SMA(RSV, k); RSV = (C - Low_n) / (High_n - Low_n) * 100'
    },
}

indicators_technical_other = {
    'Kaufman Adaptive Moving Average': {
        'sub-category': 'moving average',
        'data': ['close'],
        'function': _calc_KAMA,
        'note': 'ER period 10, fast period 2, slow period 30; Require 60 data points to warm up',
        'data_lookback_n': lambda n_slow=30, **kwargs: n_slow * 2,
        'formula': r'ER_t = |C_t - C_{t-n_{er}}| / \sum_{i=0}^{n_{er}-1} |C_{t-i} - C_{t-i-1}|; SC_t = (ER_t * (2/(n_{fast}+1) - 2/(n_{slow}+1)) + 2/(n_{slow}+1))^2; KAMA_t = KAMA_{t-1} + SC_t * (C_t - KAMA_{t-1})'
    },
    'MACD Line': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            MACD_Line=lambda df: df['EMA_fast'] - df['EMA_slow'],
        )['MACD_Line'],
        'note': 'Standard parameters: fast 12, slow 26; Require 104 data points to warm up; EMA uses the first data point as the initial value',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'MACD Line = EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)'
    },
    'MACD Signal': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26, n_signal=9: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            MACD_Line=lambda df: df['EMA_fast'] - df['EMA_slow'],
            Signal_Line=lambda df: df['MACD_Line'].ewm(span=n_signal, min_periods=n_signal, adjust=False).mean(),
        )['Signal_Line'],
        'note': 'Standard parameters: fast 12, slow 26, signal 9; Require 104 data points to warm up; EMA uses the first data point as the initial value',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'Signal Line = EMA_{n_{signal}}(MACD Line); MACD Line = EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)'
    },
    'MACD Histogram': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26, n_signal=9: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            MACD_Line=lambda df: df['EMA_fast'] - df['EMA_slow'],
            Signal_Line=lambda df: df['MACD_Line'].ewm(span=n_signal, min_periods=n_signal, adjust=False).mean(),
            MACD_Hist=lambda df: df['MACD_Line'] - df['Signal_Line']
        )['MACD_Hist'],
        'note': 'Standard parameters: fast 12, slow 26, signal 9; Require 104 data points to warm up; EMA uses the first data point as the initial value',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'MACD Histogram = MACD Line - Signal Line; MACD Line = EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P); Signal Line = EMA_{n_{signal}}(MACD Line)'
    },
    'Parabolic SAR': {
        'sub-category': 'trend',
        'data': ['high', 'low', 'close'],
        'function': _calc_SAR,
        'note': 'Initial AF 0.02, step 0.02, max 0.20; Require 10 data points to warm up',
        'data_lookback_n': lambda **kwargs: 10,
        'formula': 'SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1}); EP is the extreme point of the current trend (high for long, low for short); AF starts at initial_af, accelerates by step_af each new extreme until max_af; on reversal SAR resets to the prior trend EP and AF resets to initial_af'
    },
    'Stochastic RSI': {
        'sub-category': 'momentum',
        'data': ['close'],
        'function': lambda df, n=14, k=14, k_smooth=3: df.assign(
            Change=lambda df: df['close'].diff(),
            Gain=lambda df: df['Change'].clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean(),
            Loss=lambda df: -df['Change'].clip(upper=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean(),
            RSI=lambda df: 100 - (100 / (1 + (df['Gain'] / df['Loss']))),
            MinRSI=lambda df: df['RSI'].rolling(window=k).min(),
            MaxRSI=lambda df: df['RSI'].rolling(window=k).max(),
            StochRSI=lambda df: (df['RSI'] - df['MinRSI']) / (df['MaxRSI'] - df['MinRSI']),
            StochRSI_Smoothed=lambda df: df['StochRSI'].rolling(window=k_smooth).mean()
        )['StochRSI_Smoothed'],
        'note': 'Standard parameters: RSI period 14, Stoch period 14, smooth 3; Gain/Loss calculation uses Wilder smoothing, using the first data point as the initial value; output smoothed with SMA; Require 73 data points to warm up',
        'data_lookback_n': lambda n=14, k=14, k_smooth=3, **kwargs: n * 4 + k + k_smooth,
        'formula': 'StochRSI_Smoothed = SMA_k(StochRSI); StochRSI = (RSI - min(RSI, k)) / (max(RSI, k) - min(RSI, k)) * 100; RSI = 100 - (100 / (1 + Gain / Loss)); Gain = Wilder_n(max(Change, 0)); Loss = Wilder_n(-min(Change, 0))'
    },
    'Percentage Price Oscillator': {
        'sub-category': 'trend',
        'data': ['close'],
        'function': lambda df, n_fast=12, n_slow=26: df.assign(
            EMA_fast=lambda df: df['close'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean(),
            EMA_slow=lambda df: df['close'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean(),
            PPO=lambda df: (df['EMA_fast'] - df['EMA_slow']) / df['EMA_slow'] * 100
        )['PPO'],
        'note': 'Standard parameters: fast 12, slow 26; Require 104 data points to warm up; EMA uses the first data point as the initial value',
        'data_lookback_n': lambda n_slow=26, **kwargs: n_slow * 4,
        'formula': 'PPO = (EMA_{n_{fast}}(P) - EMA_{n_{slow}}(P)) / EMA_{n_{slow}}(P) * 100'
    },
    'True Range': {
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
    'Chaikin Volatility': {
        'sub-category': 'volatility',
        'data': ['high', 'low'],
        'function': lambda df, n=10, m=10: df.assign(
            HL_EMA=lambda df: (df['high'] - df['low']).ewm(span=n, min_periods=n, adjust=False).mean(),
            ChaikinVol=lambda df: (df['HL_EMA'] / df['HL_EMA'].shift(m)) - 1
        )['ChaikinVol'],
        'note': 'Standard parameters: (10, 10); Require 50 bars to warm up; EMA uses the first data point as the initial value',
        'data_lookback_n': lambda n=10, m=10: n * 4 + m,
        'formula': 'HL_EMA_t = EMA_n(H - L); ChaikinVol_t = HL_EMA_t / HL_EMA_{t-m} - 1'
    },
    'On-Balance Volume': {
        'sub-category': 'volume',
        'data': ['close', 'volume'],
        'function': lambda df: (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum(),
        'note': 'Requires 60 bars to warm up',
        'data_lookback_n': lambda **kwargs: 60,
        'formula': r'OBV_t = OBV_{t-1} + sign(C_t - C_{t-1}) * V_t = \sum_{i=1}^{t} sign(C_i - C_{i-1}) * V_i'
    },
    'Force Index': {
        'sub-category': 'volume',
        'data': ['close', 'volume'],
        'function': lambda df: df['close'].diff() * df['volume'],
        'data_lookback_n': lambda **kwargs: 1,
        'formula': 'ForceIndex_t = (C_t - C_{t-1}) * V_t'
    },
    'Accumulation/Distribution Line': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df: df.assign(
            MFM=lambda df: ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan),
            MFV=lambda df: df['MFM'] * df['volume'],
            ADL=lambda df: df['MFV'].cumsum()
        )['ADL'],
        'note': 'Requires 60 bars to warm up',
        'data_lookback_n': lambda **kwargs: 60,
        'formula': r'MFM_t = ((C_t - L_t) - (H_t - C_t)) / (H_t - L_t); MFV_t = MFM_t * V_t; ADL_t = ADL_{t-1} + MFV_t = \sum_{i=1}^{t} MFV_i'
    },
    'Chaikin Money Flow': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df, n=20: df.assign(
            MFM=lambda df: ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan),
            MFV=lambda df: df['MFM'] * df['volume'],
            CMF=lambda df: df['MFV'].rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
        )['CMF'],
        'note': 'Standard period: 20',
        'data_lookback_n': lambda n=20, **kwargs: n - 1,
        'formula': r'MFM_t = ((C_t - L_t) - (H_t - C_t)) / (H_t - L_t); MFV_t = MFM_t * V_t; CMF = \sum_{i=0}^{n-1} MFV_{t-i} / \sum_{i=0}^{n-1} V_{t-i}'
    },
    'Chaikin Oscillator': {
        'sub-category': 'volume',
        'data': ['high', 'low', 'close', 'volume'],
        'function': lambda df, n_fast=3, n_slow=10: df.assign(
            MFM=lambda df: ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan),
            MFV=lambda df: df['MFM'] * df['volume'],
            ADL=lambda df: df['MFV'].cumsum(),
            ChaikinOsc=lambda df: df['ADL'].ewm(span=n_fast, min_periods=n_fast, adjust=False).mean() - df['ADL'].ewm(span=n_slow, min_periods=n_slow, adjust=False).mean()
        )['ChaikinOsc'],
        'note': 'Standard parameters: fast 3, slow 10; Require 60 bars to warm up; EMA uses the first data point as the initial value',
        'data_lookback_n': lambda **kwargs: 60,
        'formula': 'ChaikinOsc = EMA_{n_{fast}}(ADL) - EMA_{n_{slow}}(ADL)'
    },
}

indicators_technical_performance = {
    'Annualized Return': {
        'sub-category': 'return',
        'data': ['close'],
        'function': lambda df: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (252 / (len(df) - 1)) - 1) * 100,
        'note': '252 trading days per year; unit: %',
        'formula': '((P_T / P_0)^{252 / N} - 1) * 100%'
    },
    'Annualized Volatility': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: df['close'].pct_change().dropna().std() * np.sqrt(252) * 100,
        'note': '252 trading days per year; unit: %',
        'formula': r'r_t = P_t / P_{t-1} - 1; AnnVol = \sigma(r) * \sqrt{252} * 100%'
    },
    'Skewness': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: df['close'].pct_change().dropna().skew(),
        'formula': r'Skew(r) = E[(r - \mu)^3] / \sigma^3'
    },
    'Excess Kurtosis': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: df['close'].pct_change().dropna().kurtosis(),
        'formula': r'Kurt(r) = E[(r - \mu)^4] / \sigma^4 - 3'
    },
    'Downside Deviation': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, rf=0: np.sqrt(np.mean(np.minimum(df['close'].pct_change().dropna() - rf, 0)**2)),
        'note': 'Default target return is 0; not annualized',
        'formula': r'DD = \sqrt{1 / N * \sum_{t=1}^{N} min(r_t - rf, 0)^2}'
    },
    'Maximum Drawdown': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: (1 - df['close'] / df['close'].cummax()).max() * 100,
        'note': 'Reported as a positive number; unit: %',
        'formula': r'MDD = max_{t} (1 - P_t / max_{s \le t} P_s) * 100%'
    },
    'Tracking Error': {
        'sub-category': 'risk',
        'data': ['close', 'benchmark_close'],
        'function': lambda df: (df['close'].pct_change() - df['benchmark_close'].pct_change()).dropna().std() * np.sqrt(252),
        'note': 'Annualized; 252 trading days per year',
        'formula': r'TE = \sigma(r - r_b) * \sqrt{252}; r_b is the benchmark daily return'
    },
    'Information Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'benchmark_close'],
        'function': lambda df: (
            lambda df_ret: (df_ret['close'] - df_ret['benchmark_close']).mean() * 252 / ((df_ret['close'] - df_ret['benchmark_close']).std() * np.sqrt(252))
        )(df[['close', 'benchmark_close']].pct_change().dropna()),
        'note': 'Annualized (US standard uses arithmetic mean excess return); 252 trading days per year',
        'formula': r'IR = E[r - r_b] * 252 / TE; TE = \sigma(r - r_b) * \sqrt{252}; r_b is the benchmark daily return'
    },
    'Sharpe Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df, rf=0: (df['close'].pct_change().dropna().mean() * 252 - rf) / (df['close'].pct_change().dropna().std() * np.sqrt(252)),
        'note': 'Default risk-free rate is 0; annualized (US standard uses arithmetic mean return); 252 trading days per year',
        'formula': r'SR = (AnnRet - rf) / (\sigma(r) * \sqrt{252})'
    },
    'Sortino Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df, rf=0: (df['close'].pct_change().dropna().mean() * 252 - rf) / np.sqrt(np.mean(np.minimum(df['close'].pct_change().dropna() - rf/252, 0)**2) * 252),
        'note': 'Default target return is 0; annualized (US standard uses arithmetic mean return); 252 trading days per year',
        'formula': r'Sortino = (AnnRet - rf) / \sqrt{DD^2 * 252}'
    },
    'Calmar Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (252 / (len(df) - 1)) - 1) / (1 - df['close'] / df['close'].cummax()).max(),
        'note': 'Return annualized (using CAGR); 252 trading days per year',
        'formula': 'Calmar = AnnRet / MDD'
    },
    'Treynor Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'market_close'],
        'function': lambda df, rf=0: (df['close'].pct_change().dropna().mean() * 252 - rf) / df['close'].pct_change().dropna().cov(df['market_close'].pct_change().dropna()) * df['market_close'].pct_change().dropna().var(),
        'note': 'Default risk-free rate is 0; return annualized (US standard uses arithmetic mean return); 252 trading days per year',
        'formula': r'Treynor = (AnnRet - rf) / \beta; \beta = cov(r, r_m) / var(r_m)'
    },
    'Modigliani Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'benchmark_close'],
        'function': lambda df, rf=0: rf + (df['close'].pct_change().dropna().mean() * 252 - rf) / df['close'].pct_change().dropna().std() * df['benchmark_close'].pct_change().dropna().std(),
        'note': 'Default risk-free rate is 0; return annualized (US standard uses arithmetic mean return); 252 trading days per year',
        'formula': r'M^2 = rf + (AnnRet - rf) / \sigma(r) * \sigma(r_b)'
    },
    'Ulcer Index': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df: ((1 - df['close'] / df['close'].cummax()) ** 2).mean() ** 0.5,
        'formula': r'UI = \sqrt{1 / N * \sum_{t=1}^{N} (1 - P_t / max_{s \le t} P_s)^2}'
    },
    'Martin Ratio': {
        'sub-category': 'risk-adjusted return',
        'data': ['close'],
        'function': lambda df: ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (252 / (len(df) - 1)) - 1) / (((1 - df['close'] / df['close'].cummax()) ** 2).mean() * 252) ** 0.5 ,
        'note': 'Numerator and denominator both annualized; 252 trading days per year',
        'formula': r'Martin = AnnRet / \sqrt{UI^2 * 252}'
    },
    'Beta': {
        'sub-category': 'risk',
        'data': ['close', 'market_close'],
        'function': lambda df: df['close'].pct_change().dropna().cov(df['market_close'].pct_change().dropna()) / df['market_close'].pct_change().dropna().var(),
        'formula': r'\beta = cov(r, r_m) / var(r_m)'
    },
    "Jensen's Alpha": {
        'sub-category': 'risk-adjusted return',
        'data': ['close', 'market_close'],
        'function': lambda df, rf=0: (
            lambda df_ret: (df_ret['close'].mean() * 252) - (rf + (df_ret['close'].cov(df_ret['market_close']) / df_ret['market_close'].var()) * ((df_ret['market_close'].mean() * 252) - rf))
        )(df[['close', 'market_close']].pct_change().dropna()),
        'note': 'Default risk-free rate is 0; return annualized (US standard uses arithmetic mean return); 252 trading days per year',
        'formula': r'\alpha = AnnRet - [rf + \beta * (AnnRet_m - rf)]; \beta = cov(r, r_m) / var(r_m)'
    },
    'VaR (Parametric)': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: (
            lambda ret: norm.ppf(confidence) * ret.std() - ret.mean()
        )(df['close'].pct_change().dropna()),
        'note': '1-day VaR; loss expressed as a positive number; assumes normality; default confidence 95%',
        'formula': r'VaR_c = \Phi^{-1}(c) * \sigma(r) - \mu(r)'
    },
    'VaR (Historical Simulation)': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: -df['close'].pct_change().dropna().quantile(1 - confidence),
        'note': '1-day VaR; loss expressed as a positive number; default confidence 95%',
        'formula': 'VaR_c = -Q_{1-c}(r); Q_p is the p-th quantile'
    },
    'Expected Shortfall (Parametric)': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: (
            lambda ret: norm.pdf(norm.ppf(confidence)) / (1 - confidence) * ret.std() - ret.mean()
        )(df['close'].pct_change().dropna()),
        'note': '1-day ES; assumes normality; default confidence 95%',
        'formula': r'ES_c = \phi(\Phi^{-1}(c)) / (1 - c) * \sigma(r) - \mu(r)'
    },
    'Expected Shortfall (Historical Simulation)': {
        'sub-category': 'risk',
        'data': ['close'],
        'function': lambda df, confidence=0.95: (
            lambda ret: -ret[ret <= ret.quantile(1 - confidence)].mean()
        )(df['close'].pct_change().dropna()),
        'note': '1-day ES; default confidence 95%',
        'formula': r'ES_c = -E[r | r \le Q_{1-c}(r)]'
    },
}


indicators_macro = {
    'Incremental Capital Output Ratio': {
        'sub-category': 'productivity',
        'data': ['Real Gross Private Domestic Investment (Billion USD)', 'Real GDP (Billion USD)'],
        'function': lambda df: df['Real Gross Private Domestic Investment (Billion USD)'] / df['Real GDP (Billion USD)'].diff(),
        'formula': r'ICOR = \Delta K / \Delta Y = Real Gross Private Domestic Investment / change in GDP',
        'note': 'Use BEA real investment and real GDP (chained 2017 dollars).'
    },
    'Total Factor Productivity Growth': {
        'sub-category': 'productivity',
        'data': ['Nominal GDP (Billion USD)', 'Real GDP Growth Rate (%)', 'Total Employees (Thousand)', 'Compensation of Employees (Billion USD)'],
        'function': lambda df: df.assign(
            Delta_Y=lambda df: df['Real GDP Growth Rate (%)'] / 100,
            Delta_L=lambda df: df['Total Employees (Thousand)'].pct_change(),
            beta=lambda df: df['Compensation of Employees (Billion USD)'] / df['Nominal GDP (Billion USD)'],
            alpha=lambda df: 1 - df['beta'],
            Delta_TFP=lambda df: (df['Delta_Y'] - df['alpha'] * 0.06 - df['beta'] * df['Delta_L']) * 100,
         )['Delta_TFP'],
        'formula': r'TFP Growth = (\Delta Y - \alpha \Delta K - \beta \Delta L) * 100%; \alpha = 1 - \beta; \beta = Compensation of Employees / GDP',
        'note': r'Unit: %; \Delta K is set to a constant 6%'
    },
    'Trade Openness': {
        'sub-category': 'external trade',
        'data': ['Exports of Goods and Services (Billion USD)', 'Imports of Goods and Services (Billion USD)', 'Nominal GDP (Billion USD)'],
        'function': lambda df: (df['Exports of Goods and Services (Billion USD)'] + df['Imports of Goods and Services (Billion USD)']) / df['Nominal GDP (Billion USD)'] * 100,
        'formula': '(Exports + Imports) / GDP * 100%',
        'note': 'Unit: %'
    },
    'Federal Debt to GDP': {
        'sub-category': 'fiscal',
        'data': ['Gross Federal Debt Held by the Public (Billion USD)', 'Nominal GDP (Billion USD)'],
        'function': lambda df: df['Gross Federal Debt Held by the Public (Billion USD)'] / df['Nominal GDP (Billion USD)'] * 100,
        'formula': 'Federal Debt Held by the Public / GDP * 100%',
        'note': 'Unit: %'
    },
    'Fiscal Deficit Ratio': {
        'sub-category': 'fiscal',
        'data': ['Federal Outlays (Billion USD)', 'Federal Receipts (Billion USD)', 'Nominal GDP (Billion USD)'],
        'function': lambda df: (df['Federal Outlays (Billion USD)'] - df['Federal Receipts (Billion USD)']) / df['Nominal GDP (Billion USD)'] * 100,
        'formula': 'Federal Deficit / GDP * 100%; Federal Deficit = Federal Outlays - Federal Receipts',
        'note': 'Unit: %'
    },
    'Federal Tax Burden': {
        'sub-category': 'fiscal',
        'data': ['Federal Tax Receipts (Billion USD)', 'Nominal GDP (Billion USD)'],
        'function': lambda df: df['Federal Tax Receipts (Billion USD)'] / df['Nominal GDP (Billion USD)'] * 100,
        'formula': 'Federal Tax Receipts / GDP * 100%',
        'note': 'Unit: %'
    },
    'Macro Real Rate': {
        'sub-category': 'inflation',
        'data': ['10-Year Treasury Constant Maturity Rate (%)', 'GDP Deflator Inflation (%)'],
        'function': lambda df: df['10-Year Treasury Constant Maturity Rate (%)'] - df['GDP Deflator Inflation (%)'],
        'formula': '10-Year Treasury Constant Maturity Rate - GDP Deflator Inflation',
        'note': 'Unit: %; Use 10-Year Treasury rate as the risk-free rate; use GDP deflator for inflation'
    },
    'Household Real Rate': {
        'sub-category': 'inflation',
        'data': ['30-Year Fixed Rate Mortgage Average (%)', 'CPI Inflation (%)'],
        'function': lambda df: df['30-Year Fixed Rate Mortgage Average (%)'] - df['CPI Inflation (%)'],
        'formula': '30-Year Fixed Rate Mortgage Average - CPI Inflation',
        'note': 'Unit: %; Use 30-Year Fixed Mortgage rate as a proxy for household borrowing cost; use CPI for consumer inflation'
    },
    'Corporate Real Rate': {
        'sub-category': 'inflation',
        'data': ["Moody's Seasoned Baa Corporate Bond Yield", 'GDP Deflator Inflation (%)'],
        'function': lambda df: df["Moody's Seasoned Baa Corporate Bond Yield"] - df['GDP Deflator Inflation (%)'],
        'formula': "Moody's Seasoned Baa Corporate Bond Yield - GDP Deflator Inflation",
        'note': "Unit: %; Use Moody's Baa Corporate Bond Yield as a proxy for corporate borrowing cost; use GDP deflator for inflation"
    },
    'Real Wage Growth Rate': {
        'sub-category': 'labor',
        'data': ['Average Hourly Earnings (USD)', 'CPI Inflation (%)'],
        'function': lambda df: df['Average Hourly Earnings (USD)'].pct_change() * 100 - df['CPI Inflation (%)'],
        'formula': 'Average Hourly Earnings Growth Rate - CPI Inflation',
        'note': 'Unit: %; Use Average Hourly Earnings as a proxy for wage'
    }
}
