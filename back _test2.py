from function2 import *
import pymssql
import warnings
import joblib
import glob
import os
import shutil
warnings.filterwarnings("ignore")

# 清理储存回测结果的文件夹
shutil.rmtree('model_output')
os.mkdir('model_output')

# 加载文件名称
file_names = sorted(glob.glob('dataset/factorPool_PV_20230303/因子中性化数据0707/*'))

#加载模型
model_names = sorted(glob.glob('model/*'))

# 保持和模型训练部分一致
time_unit = '月'   # 数据长度单位
unit = '月'   # 收益频率
n_train, n_test = 12, 12   # 训练集和测试集长度
n_evaluate = 12   # 验证集长度
response_type = '收益率'

'''分组回测'''

conn = pymssql.connect('(local)', 'sa', '123456', 'winddb0501')  # 连接sql server

for model_path in model_names:
    # 选择模型
    model_time = model_path[6:16]
    model = joblib.load(model_path)
    OutputPath = 'model_output/{}_model'.format(model_time)
    os.mkdir(OutputPath)

    print(model_time + '模型回测开始')
    # ---数据部分---

    print('读取数据')
    Calender = Load_Calender(file_names, model_time, n_evaluate, time_unit, '日')  # 加载数据日期
    Risk_free_file = 'Input/无风险利率.xlsx'  # 无风险利率数据路径
    RiskFreeReturn = LoadRiskFreeReturn(Risk_free_file, Calender)  # 加载无风险利率数据
    Price, MarketCap, Industry, ST, Suspend, ListDate = Load_SQLData(conn, Calender, False, OutputPath,
                                                                     local=False)  # 加载SQL数据
    BenchReturn = Load_Return(conn, Calender)  # 加载基准收益
    back_test_files = filter_file_name(file_names, Calender)  # 加载因子数据文件
    back_test_x, back_test_y = read_data(back_test_files, response_type, unit)
    back_test_x = Get_FACTOR_Neu(back_test_x, model)  # 加载模型预测值

    # ---分组部分---

    GroupNum = 10  # 分组数量
    Factor_Group = Cal_Stratify(back_test_x, GroupNum)  # 加载分组

    # ---回测部分---
    Perform = pd.DataFrame(index=['分组' + str(i) for i in range(1, GroupNum + 1)],
                           columns=['年化收益(%)', '基准年化收益(%)', '超额年化收益(%)', '年化波动(%)',
                                    '基准年化波动(%)',
                                    '超额年化波动(%)', '最大回撤(%)', '基准最大回撤(%)', '超额最大回撤(%)', '夏普比率',
                                    '基准夏普比率', '信息比率', '收益回撤比', '基准收益回撤比', '超额收益回撤比',
                                    '胜率(%)',
                                    '换手率(年均)'])
    NetValue = pd.DataFrame(index=back_test_x['TRADE_DT'].unique(), columns=['基准净值'])
    writer = pd.ExcelWriter(OutputPath + '/分组回测结果.xlsx')
    for i in tqdm(range(1, GroupNum + 1),desc='分组回测...'):
        # 获取分组持仓的股票池
        Hold = Factor_Group[Factor_Group['GROUP'] == str(i)][['TRADE_DT', 'S_INFO_WINDCODE']].reset_index(drop=True)
        # 根据持仓股票池，生成仓位数据
        Position = Cal_Position(Hold)
        Position.to_excel(writer, sheet_name='第' + str(i) + '组仓位')
        # 调用回测函数
        NetValueTmp, PerformTmp = Back_Testing(Position, Price, BenchReturn, RiskFreeReturn)
        Perform.loc['分组' + str(i)] = PerformTmp.loc['绩效统计']
        NetValue['基准净值'] = NetValueTmp['基准净值']
        NetValue['分组' + str(i) + '净值'] = NetValueTmp['策略净值']
        NetValue['分组' + str(i) + '相对净值'] = NetValueTmp['相对净值']

    Perform.to_excel(writer, sheet_name='绩效统计')
    NetValue.to_excel(writer, sheet_name='净值统计')
    writer.save()
    writer.close()



