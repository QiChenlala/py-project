import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm

# ------数据部分------
# 加载文件

def is_legal(filename, start_date, end_date):
    '''
    函数名称：is_legal
    函数功能：根据数据名称判断数据是否在规定区间内
    输入参数：filename：数据名称列表;
            start_date, end_date: 设置过滤数据的区间,只包含开始日;
    输出参数：True/False
    '''
    date = pd.to_datetime(filename[43:-4])
    return (date >= pd.to_datetime(start_date) and date < pd.to_datetime(end_date))

def Load_Calender(file_names, model_date, n, time_unit, data_interval_unit):
    '''
    函数名称：Load_Calender
    函数功能：读取日历
    输入参数：filename：数据名称列表；
            model_date：日历开始日期；
            n：日历整体长度；
            time_unit：日历整体长度的单位；
            data_interval_unit：日历中抽取日期的频率
    输出参数：calender
    '''
    calender_list = list(pd.to_datetime(x[43:-4]) for x in file_names if is_legal(x, model_date, str(pd.to_datetime(model_date) + time_interval(n, time_unit))))
    start_date = pd.to_datetime(min(calender_list))
    if data_interval_unit == '周':
        result = list((x.strftime("%Y%m%d")) for x in calender_list if x.dayofweek == start_date.dayofweek)
    elif data_interval_unit == '月':
        result = list((x.strftime("%Y%m%d")) for x in calender_list if x.day == start_date.day)
    elif data_interval_unit == '年':
        result = list((x.strftime("%Y%m%d")) for x in calender_list if x.dayofyear == start_date.dayofyear)
    else:
        result = list((x.strftime("%Y%m%d")) for x in calender_list)
    calender = pd.DataFrame()
    calender['TRADE_DT'] = result
    return calender


def filter_file_name(input_file_names, Calender):
    '''
    函数名称：filter_file_name
    函数功能：根据日历过滤数据名称
    输入参数：input_file_names：数据名称列表；
            Calender：日历；
    输出参数：data
    '''
    data = list(x for x in input_file_names if x[43:-4].replace('.','') in list(Calender['TRADE_DT']))
    return data

def read_data(input_file_names, response_type, unit):
    '''
    函数名称：read_data
    函数功能：读取数据并返回特征和应变量
    输入参数：input_file_names：数据名称列表；
            response_type：应变量选择；
            unit：收益频率；
    输出参数：features，response
    '''
    units = {
        '周':5,
        '月':20
    }
    response_types = {
        '夏普率':'sharp_momentum_{}p'.format(units[unit]),
        '收益率': 'zdf{}'.format(units[unit])
    }

    result = pd.DataFrame()
    for files in tqdm(input_file_names, desc='读取数据...'):
        data = pd.read_csv(files, sep=',',encoding = 'gb2312')
        data.replace(['-0w','0w'],np.NaN, inplace=True)
        data.fillna(data.median(), inplace=True)
        data.fillna(0, inplace=True)
        for columns in ['Amount_20D_AVG', 'ep_ts_score120', 'grossprofit_qfa', 'net_profit_excl_min_int_inc_qfa', 'oper_rev_qfa_tb', 'delta_EPFY1_20d', 'np_fy1_tb_120d', 'net_profit_after_ded_nr_lp_qfa', 'net_profit_after_ded_nr_lp_qfa_tb']:
            data[columns] = data[columns].astype(float)
        result = result.append(data, ignore_index = True)
    result.rename(columns={'trade_dt': 'TRADE_DT', 's_info_windcode': 'S_INFO_WINDCODE'}, inplace=True)
    result['TRADE_DT'] = list(x.replace('-','') for x in result['TRADE_DT'])
    features = result.drop(['sharp_momentum_5p', 'sharp_momentum_20p', 'zdf5', 'zdf20'], axis = 1)

    response = result[response_types[response_type]]

    return features, response

def time_interval(n, unit):
    time_intervals = {'日':DateOffset(days=n),
                      '周':DateOffset(weeks=n),
                      '月':DateOffset(months=n),
                      '年':DateOffset(years=n)}
    return time_intervals[unit]

def valid_modle_date(file_names, model_time, train_interval, test_interval, evaluate_interval, unit):
    '''
    函数名称：valid_modle_date
    函数功能：检查是否有足够的数据训练模型
    输入参数：file_names：训练模型所需的所有文件名;
            model_time：模型的训练时间；
            train_interval：模型训练数据长度
            test_interval：模型测试数据长度
            evaluate_interval：回测数据长度
            unit：模型使用数据长度单位;
    输出参数：True/False
    '''
    all_time = list(pd.to_datetime(x[43:-4]) for x in file_names)
    start_time = pd.to_datetime(model_time) - time_interval(train_interval+test_interval, unit)
    end_time = pd.to_datetime(model_time) + time_interval(evaluate_interval, unit)
    return (end_time <= max(all_time)) and (start_time >= min(all_time))

def Cal_Stratify(data, GroupNum):
    '''
    函数名称：Cal_Stratify
    函数功能：获取因子值分组
    输入参数：data：因子值；GroupNum：组数
    输出参数：Result：因子分组
    '''
    # 按照ep分成10组，分别计算收益率
    data_Rank = data.groupby('TRADE_DT').apply(Get_Rank).reset_index(drop=True)
    length = data_Rank.groupby('TRADE_DT')[['RANK']].count()  # 计算每月共有多少只股票
    # bins记录分组的边界，是分组的依据（左闭右开）
    bins = pd.DataFrame(length['RANK'].apply(lambda _: [math.ceil(_ / GroupNum * i) for i in range(GroupNum + 1)]))
    GroupLabels = [str(i) for i in range(1, GroupNum + 1)]  # 为每组标号
    data_Rank = data_Rank.groupby('TRADE_DT').apply(Get_Group, bins, GroupLabels).reset_index(drop=True)
    Result = data_Rank[['TRADE_DT', 'S_INFO_WINDCODE', 'GROUP']]
    return Result

def Get_Rank(group):
    '''
    函数名称：Get_Rank
    函数功能：排序，groupby内部函数
    输入参数：group：待排序值
    输出参数：group：添加了一列序号
    '''
    group['RANK'] = group[['predicted_y']].rank(ascending=False, method='first', axis=0, na_option='keep')
    return group


def Get_Group(group, bins, GroupLabels):
    '''
    函数名称：Get_Group
    函数功能：分组，groupby内部函数
    输入参数：group：待排序值；bins分组边界；GroupLabels：各组标签
    输出参数：group：添加了一列组标签
    '''
    group['GROUP'] = pd.cut(group.RANK, bins.loc[group.TRADE_DT.values[0], 'RANK'], labels=GroupLabels)
    return group

def Get_FACTOR_Neu(input_data, model):
    result = input_data
    result['predicted_y'] = model.predict(input_data.drop(['TRADE_DT', 'S_INFO_WINDCODE'], axis = 1))
    return result


def Cal_Position(Hold):
    '''
    函数名称：Cal_Position
    函数功能：根据持仓股票池计算下期仓位
    输入参数：Hold：持仓股票池
    输出参数：Position：仓位
    '''
    PeriodList=Hold.TRADE_DT.unique()
    Position = Hold.copy()
    Position['POSITION'] = 1
    # 移动至下期
    Position.TRADE_DT = [PeriodList[np.where(PeriodList==i)[0][0]+1] if np.where(PeriodList==i)[0][0]+1 < len(PeriodList) else np.nan for i in Position.TRADE_DT]
    Position.dropna(subset=['TRADE_DT'],inplace=True)

    Position['POSITION'] = Position.groupby('TRADE_DT').apply(lambda x: x['POSITION'] / sum(x['POSITION'])).reset_index(drop=True).values
    Position.loc[len(Position.index)] = [PeriodList[np.where(PeriodList==Position.TRADE_DT.values[0])[0][0]-1], 'StockCode', 0]
    Position = Position.sort_values(by=['TRADE_DT','S_INFO_WINDCODE']).reset_index(drop=True)
    return Position


def Back_Testing(Position, Price, BenchReturn, RiskFreeReturn):
    '''
    函数名称：Back_Testing
    函数功能：回测函数
    输入参数：Position：仓位数据；Price：股票价格数据；BenchReturn：基准收益率；RiskFreeReturn：无风险收益率；StartDate：起始日期；EndDate：截止日期
    输出参数：NetValue：净值；Perform：绩效统计
    '''
    PeriodList=Position.TRADE_DT.unique()
    # 计算策略的日频收益率
    Price['TRADE_PERIOD']=[PeriodList[np.where(PeriodList>=i)[0][0]] for i in Price.TRADE_DT]
    StrategyReturn = pd.DataFrame(Price.groupby(Price.TRADE_PERIOD).apply(M2D, Position).values,index=Price.TRADE_DT.unique(), columns=['RETURN'])
    StrategyReturn.fillna(0, inplace=True)
    # 计算换手率
    TurnOverRate = Cal_TurnOver(Position)
    # 计算无风险利率（货基指数）的年化收益
    RiskfreeReturn, RiskfreeIntervalRet, RiskfreeIntervaStd, RiskfreeAnnualRet, RiskfreeAnnualStd, RiskfreeMaxdrawdown = PerfStatis(RiskFreeReturn, 'D')
    # 统计策略的净值、区间收益率、区间波动率、年化收益率、年化波动率、夏普比率、最大回撤
    StrategyReturn, StrategyIntervalRet, StrategyIntervaStd, StrategyAnnualRet, StrategyAnnualStd, StrategyMaxdrawdown = PerfStatis(StrategyReturn, 'D')
    # 统计基准的净值、区间收益率、区间波动率、年化收益率、年化波动率、夏普比率、最大回撤
    BenchReturn.loc[BenchReturn.index<=PeriodList[0],'RETURN']=0
    BenchReturn, BenchIntervalRet, BenchIntervaStd, BenchAnnualRet, BenchAnnualStd, BenchMaxdrawdown = PerfStatis(BenchReturn, 'D')
    # 统计超额的净值、区间收益率、区间波动率、年化收益率、年化波动率、夏普比率、最大回撤
    ExcessReturn = Cal_Excess(StrategyReturn, BenchReturn)
    ExcessReturn, ExcessIntervalRet, ExcessIntervaStd, ExcessAnnualRet, ExcessAnnualStd, ExcessMaxdrawdown = PerfStatis(ExcessReturn, 'D')
    StrategyWinper = Cal_Winper(ExcessReturn)

    # 保存总体绩效统计
    Perform = pd.DataFrame(index=['绩效统计'],columns=['年化收益(%)', '基准年化收益(%)', '超额年化收益(%)', '年化波动(%)', '基准年化波动(%)', '超额年化波动(%)', '最大回撤(%)','基准最大回撤(%)', '超额最大回撤(%)', '夏普比率', '基准夏普比率', '信息比率', '收益回撤比', '基准收益回撤比','超额收益回撤比', '胜率(%)', '换手率(年均)'])

    Perform['年化收益(%)'], Perform['基准年化收益(%)'], Perform['超额年化收益(%)'], Perform['年化波动(%)'], Perform['基准年化波动(%)'], Perform['超额年化波动(%)'], \
    Perform['最大回撤(%)'], Perform['基准最大回撤(%)'], Perform['超额最大回撤(%)'], Perform['收益回撤比'], Perform['基准收益回撤比'], Perform['超额收益回撤比'], Perform['胜率(%)'], Perform['换手率(年均)'] = \
        StrategyAnnualRet, BenchAnnualRet, ExcessAnnualRet, StrategyAnnualStd, BenchAnnualStd, ExcessAnnualStd, StrategyMaxdrawdown, BenchMaxdrawdown, ExcessMaxdrawdown, \
        -(StrategyAnnualRet / StrategyMaxdrawdown), -(BenchAnnualRet / BenchMaxdrawdown), -(ExcessAnnualRet / ExcessMaxdrawdown), StrategyWinper, TurnOverRate
    Perform['夏普比率'] = (Perform['年化收益(%)'] - RiskfreeAnnualRet) / Perform['年化波动(%)']
    Perform['基准夏普比率'] = (Perform['基准年化收益(%)'] - RiskfreeAnnualRet) / Perform['基准年化波动(%)']
    Perform['信息比率'] = (Perform['超额年化收益(%)'] - RiskfreeAnnualRet) / Perform['超额年化波动(%)']

    # 保存净值数据
    NetValue = pd.DataFrame(index=BenchReturn.index, columns=['基准净值', '策略净值', '相对净值'])
    NetValue['基准净值'] = BenchReturn['NETVALUE']
    NetValue['策略净值'] = StrategyReturn['NETVALUE']
    NetValue['相对净值'] = ExcessReturn['NETVALUE']

    return NetValue, Perform

def M2D(group, Position):
    '''
    函数名称：M2D
    函数功能：Backtest的内部函数，将月频策略转为日频收益率
            首先需要对资产的日频收益率进行再平衡：根据月频仓位计算每个月内的各资产日频净值，再反推每个月内的各资产日频收益率，以进行仓位再平衡（若只有一个资产，这一步不改变任何值，有多个资产时才起作用）
    输入参数：group：单月的日频各标的收益率；Position：仓位
    输出参数：Result：仓位对应的日频收益率
    '''
    Period = group.TRADE_PERIOD.values[0]
    PositionPeriod = Position[Position.TRADE_DT == Period]
    NetValueTmp = pd.DataFrame(index=group.TRADE_DT.unique(), columns=['NETVALUE'])
    group = pd.merge(group[['TRADE_DT', 'S_INFO_WINDCODE', 'RETURN']], PositionPeriod[['S_INFO_WINDCODE', 'POSITION']],on='S_INFO_WINDCODE', how='right')
    # 对于异常情况（月底买入但次月停牌，即次月无收益率，则填0）
    group['TRADE_DT'].fillna(group.TRADE_DT.unique()[0], inplace=True)
    group['RETURN'].fillna(0, inplace=True)

    if (group['POSITION'] == 0).all():
        NetValueTmp.loc[:] = np.ones([len(NetValueTmp), 1])
    else:
        PositionValue = pd.pivot(group, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION').iloc[0].values
        groupPivot = pd.pivot(group, index='TRADE_DT', columns='S_INFO_WINDCODE', values='RETURN')
        NetValueTmp['NETVALUE'] = (PositionValue * np.cumprod(groupPivot + 1)).sum(axis=1).values

    NetValueTmp['RETURN'] = NetValueTmp['NETVALUE'] / NetValueTmp['NETVALUE'].shift(1) - 1
    # 第一天的组合收益率即第一天所有股票收益率均值
    NetValueTmp['RETURN'].values[0] = group[group.TRADE_DT == min(group.TRADE_DT)]['RETURN'].mean()
    Result = NetValueTmp[['RETURN']]
    return Result

def Cal_TurnOver(Position):
    '''
    函数名称：Cal_TurnOver
    函数功能：计算换手率
    输入参数：Position：仓位数据
    输出参数：TurnOverRate：换手率
    '''
    Position_Pivot = pd.pivot(Position, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION')
    Position_Pivot.fillna(0, inplace=True)
    TurnOverRate = np.sum(abs(Position_Pivot.values - Position_Pivot.shift(1).values), axis=1)
    TurnOverRate[0] = 0
    TurnOverRate = np.sum(TurnOverRate) / (len(TurnOverRate) / 12)
    return TurnOverRate


def PerfStatis(Return, frequency):
    '''
    函数名称：PerfStatis
    函数功能：Backtest的内部函数，用于统计绩效指标
    输入参数：Return：收益率；frequency：频率("M","W","D")
    输出参数：区间收益率 区间波动率 年化收益率 年化波动率 夏普比率 最大回撤
    '''
    if frequency == 'M':
        N = 12
    elif frequency == 'W':
        N = 52
    elif frequency == 'D':
        N = 250
    ET = len(Return)  # 有效长度
    # 计算净值
    Return['NETVALUE'] = np.cumprod(Return['RETURN'] + 1)
    # 计算区间收益率、区间波动率、年化收益率和年化波动率
    IntervalRet = 100 * (Return['NETVALUE'].values[-1] / Return['NETVALUE'].values[0] - 1)
    IntervaStd = 100 * np.std(Return['RETURN'].values)
    AnnualRet = 100 * ((Return['NETVALUE'].values[-1] / Return['NETVALUE'].values[0]) ** (N / ET) - 1)
    AnnualStd = 100 * np.std(Return['RETURN'].values) * np.sqrt(N)
    # 计算最大回撤
    Maxdrawdown = 100 * min(Return['NETVALUE'] / Return['NETVALUE'].cummax() - 1)
    return Return, IntervalRet, IntervaStd, AnnualRet, AnnualStd, Maxdrawdown


def Cal_Excess(StrategyReturn, BenchReturn):
    '''
    函数名称：Cal_Excess
    函数功能：计算超额收益
    输入参数：StrategyReturn：策略收益；BenchReturn：基准收益
    输出参数：ExcessReturn：超额收益
    '''
    ExcessReturn = pd.DataFrame(index=BenchReturn.index, columns=['RETURN'])
    ExcessReturn['NETVALUE'] = StrategyReturn['NETVALUE'] / BenchReturn['NETVALUE']
    ExcessReturn['RETURN'] = ExcessReturn['NETVALUE'] / ExcessReturn['NETVALUE'].shift(1) - 1
    ExcessReturn['RETURN'].values[0] = 0
    return ExcessReturn


def Cal_Winper(Return):
    '''
    函数名称：Cal_Winper
    函数功能：计算胜率
    输入参数：Return：收益率数据
    输出参数：Winper：胜率
    '''
    Winper = len(Return[Return['RETURN'] > 0]) / len(Return[Return['RETURN'] != 0])
    return Winper

def Load_SQLData(conn, Calender,test, Outputpath, local=False):
    '''
    函数名称：Load_SQLData
    函数功能：从sql server读取交易日、股票价格、市值、所属行业、ST日期、交易状态（是否停牌）、上市日期，并保存到本地pkl文件
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；local：是否从本地读取（初次运行需设为False，再次运行可设为True，节省时间）
    输出参数：Price：股票价格；MarketCap：市值；Industry：所属行业；
            ST：ST日期；Suspend：交易状态（是否停牌）；ListDate：上市日期
    '''
    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]
    # 股票价格数据
    localfile = 'Output/Price.pkl' if local else False  # 从本地读取
    Price = Load_Price(conn, StartDate, EndDate, Outputpath, localfile)
    # 市值数据
    localfile = 'Output/MarketCap.pkl' if local else False  # 从本地读取
    MarketCap = Load_MarketCap(conn, StartDate, EndDate, Outputpath, localfile)
    # 所属行业数据
    localfile = 'Output/Industry.pkl' if local else False  # 从本地读取
    Industry = Load_Industry(conn, Outputpath, localfile)
    # ST日期数据
    localfile = 'Output/ST.pkl' if local else False  # 从本地读取
    ST = Load_ST(conn, Outputpath, localfile)
    # 交易状态（是否停牌）数据
    localfile = 'Output/Suspend.pkl' if local else False  # 从本地读取
    Suspend = Load_Suspend(conn, localfile, StartDate, EndDate, Outputpath, test)
    # 上市日期数据
    localfile = 'Output/ListDate.pkl' if local else False  # 从本地读取
    ListDate = Load_ListDate(conn, Outputpath, localfile)
    MarketCap=pd.merge(Calender,MarketCap,on='TRADE_DT',how='left')
    Suspend=pd.merge(Calender,Suspend,on='TRADE_DT',how='left')

    return Price, MarketCap, Industry, ST, Suspend, ListDate

def Load_Price(conn, StartDate, EndDate, Outputpath, localfile=False):
    '''
    函数名称：Load_Price
    函数功能：从sql server读取股票价格数据
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：股票价格数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_ADJPRECLOSE,S_DQ_ADJCLOSE from AShareEODPrices where TRADE_DT>=\'" + StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/Price.pkl'.format(Outputpath))

    data.TRADE_DT = data.TRADE_DT.apply(str)
    data = data[(data.TRADE_DT >= StartDate) & (data.TRADE_DT <= EndDate)]
    data.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)
    data['RETURN'] = data['S_DQ_ADJCLOSE'] / data['S_DQ_ADJPRECLOSE'] - 1
    data = data.reset_index(drop=True)
    return data


def Load_MarketCap(conn, StartDate, EndDate, Outputpath, localfile):
    '''
    函数名称：Load_MarketCap
    函数功能：从sql server读取市值数据
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：市值数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select TRADE_DT,S_INFO_WINDCODE, S_VAL_MV from ASHAREEODDERIVATIVEINDICATOR where TRADE_DT>=\'" + StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/MarketCap.pkl'.format(Outputpath))

    data.TRADE_DT = data.TRADE_DT.apply(str)
    data = data[(data.TRADE_DT >= StartDate) & (data.TRADE_DT <= EndDate)]
    data.sort_values(by = ['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)
    data['S_VAL_MV'] = np.log(data['S_VAL_MV'] / 100)
    data = data.reset_index(drop = True)
    return data


def Load_Industry(conn, Outputpath, localfile=False):
    '''
    函数名称：Load_Industry
    函数功能：从sql server读取所属行业数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：所属行业数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql='select S_INFO_WINDCODE,substring(citics_ind_code, 1, 4) as IND_CODE, ENTRY_DT,REMOVE_DT from AShareIndustriesClassCITICS'
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/Industry.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_ST(conn, Outputpath, localfile):
    '''
    函数名称：Load_ST
    函数功能：从sql server读取ST日期数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：ST日期数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select S_INFO_WINDCODE,ENTRY_DT,REMOVE_DT from ASHAREST"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/ST.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_Suspend(conn, localfile, StartDate, EndDate, Outputpath, test):
    '''
    函数名称：Load_Suspend
    函数功能：从sql server读取交易状态（是否停牌）数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：交易状态（是否停牌）数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        if test:
            sql = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_TRADESTATUSCODE from ASHAREEODPRICES where TRADE_DT>=\'" + StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
        else:
            sql = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_TRADESTATUSCODE from ASHAREEODPRICES"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/Suspend.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_ListDate(conn, Outputpath, localfile):
    '''
    函数名称：Load_ListDate
    函数功能：从sql server读取上市日期数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：上市日期数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select S_INFO_WINDCODE,S_INFO_LISTDATE from ASHAREDESCRIPTION"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/ListDate.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_Return(conn, Calender ):

    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]

    sql = "select S_DQ_PCTCHANGE , TRADE_DT from AINDEXEODPRICES where S_INFO_WINDCODE = \'000905.SH\' AND TRADE_DT >=\'" + StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
    Return = pd.read_sql(sql, conn)
    Return.rename(columns={'S_DQ_PCTCHANGE': 'RETURN'}, inplace=True)
    Return['RETURN'] = Return['RETURN'] / 100
    Return.set_index('TRADE_DT', inplace=True)
    Return.sort_index(inplace = True)
    return Return

def LoadRiskFreeReturn(filename, Calender):
    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]

    data = pd.read_excel(filename, skiprows = 1)
    data['Date'] = list(datetime.strftime(x, '%Y%m%d') for x in data['Date'])
    Return = data[(data['Date'] >= StartDate) & (data['Date'] <= EndDate)].drop('885009.WI', axis = 1).rename(columns={'无风险利率': 'RETURN'}).set_index('Date')
    Return.sort_index(inplace = True)
    return Return







