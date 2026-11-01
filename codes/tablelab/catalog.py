"""仅注册作业指定的四个数据集。字段定义与范围声明集中维护。"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import pandas as pd

@dataclass(frozen=True)
class Spec:
    label: str
    source: str
    targets: tuple[str, ...]
    defaults: tuple[str, ...]
    excluded: tuple[str, ...]
    warning: str
    descriptions: dict[str, str]

SPECS = {
 'house': Spec('房价', 'https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data',
    ('SalePrice',), ('OverallQual','GrLivArea','YearBuilt','TotalBsmtSF','GarageCars','Neighborhood'),
    ('Id','SalePrice'),
    'SalePrice 是成交价（美元）。此处只对带标签的 train.csv 建模。Kaggle test.csv 不含成交价，不能用它计算预测误差。反复查看交互验证集后，指标不再属于独立盲测。',
    {'SalePrice':'房屋成交价，美元；连续预测目标，不是挂牌价。',
     'OverallQual':'整体材料与装修质量评分；原始编码 1 至 10。',
     'GrLivArea':'地上居住面积，平方英尺；不含地下室面积。',
     'TotalBsmtSF':'地下室总面积，平方英尺。','GarageCars':'车库容车数量。',
     'GarageArea':'车库面积，平方英尺。','Neighborhood':'所在社区，分类变量。',
     'YearBuilt':'建造年份。','YearRemodAdd':'改建年份。','YrSold':'出售年份，可用于按时间划分样本。',
     'MoSold':'出售月份。','LotArea':'地块面积，平方英尺。','LotFrontage':'临街长度，英尺。',
     'Id':'房屋记录编号；不作为预测特征。'}),
 'adult': Spec('收入分类', 'https://archive.ics.uci.edu/dataset/2/adult',
    ('income_gt_50k',), ('age','education-num','hours-per-week','workclass','occupation','marital-status'),
    ('income','income_gt_50k','fnlwgt'),
    'Adult 是从 1994 年人口普查数据库提取的样本。收入只有 >50K 与 <=50K 标签，不能计算个人工资或平均年收入。本系统默认未按 fnlwgt 加权，样本比例不能直接当作全美人口比例。原始 adult.test 留存，不参与交互模型拟合。',
    {'income_gt_50k':'派生二元变量：原始 income 为 >50K 时等于 1，<=50K 时等于 0。均值是样本中超过门槛的比例。',
     'income':'原始年收入类别。只有 >50K 与 <=50K，没有具体金额。',
     'age':'年龄，岁。','education':'教育程度分类。','education-num':'教育程度的有序数值编码，不直接等同于个人实际受教育年数。',
     'hours-per-week':'每周工作时长，小时。','workclass':'雇佣类型。','occupation':'职业类别。',
     'marital-status':'婚姻状况。','relationship':'家庭关系类别。','sex':'原数据中的性别分类。',
     'race':'原数据中的种族分类，仅用于样本统计或模型审计，不能作本质性或因果解释。',
     'capital-gain':'资本收益字段，美元；不是工资。','capital-loss':'资本损失字段，美元。',
     'native-country':'原始国家类别。','fnlwgt':'最终样本权重。当前实现不提供加权估计，不把该字段当作一般解释变量。'}),
 'nba': Spec('NBA 球员', 'https://www.kaggle.com/datasets/justinas/nba-players-data',
    ('pts','reb','ast','gp'), ('age','player_height','player_weight','draft_year'),
    ('player_name','season','Unnamed: 0'),
    '记录按球员姓名与赛季组织；同名未必同一人，缺少稳定球员 ID。不能把跨赛季重复记录当作独立个体。默认按赛季留出验证，也支持按姓名保守分组留出；不支持跨赛季自动构造滞后特征或任意未来赛季外推。同期技术统计只能用于同期估计。',
    {'player_name':'球员姓名；保守分组键，同名可能属于不同球员，不能作为唯一身份标识或预测特征。','season':'赛季，格式 YYYY-YY；时间划分使用起始年份。',
     'age':'记录的球员年龄，岁。','player_height':'球员身高，厘米。','player_weight':'球员体重，千克。',
     'pts':'赛季场均得分。','reb':'赛季场均篮板。','ast':'赛季场均助攻。','gp':'赛季出场场次。',
     'team_abbreviation':'球队简称。','draft_year':'选秀年份或 Undrafted；作为分类字段处理。',
     'draft_round':'选秀轮次或 Undrafted。','draft_number':'选秀顺位或 Undrafted。',
     'college':'大学类别。','country':'国家类别。','net_rating':'净效率统计。','usg_pct':'使用率字段。',
     'ts_pct':'真实投篮命中率字段。','ast_pct':'助攻率字段。','oreb_pct':'进攻篮板率字段。','dreb_pct':'防守篮板率字段。'}),
 'students': Spec('学生成绩', 'https://www.kaggle.com/datasets/spscientist/students-performance-in-exams',
    ('math score','reading score','writing score'),
    ('gender','parental level of education','lunch','test preparation course'), (),
    '该课程数据用于成绩分析演示。Kaggle 致谢指向 generated_data/exams 页面，不能据此当作经过核实的学生实测样本；具体生成过程未确认，不能推广为教育政策结论。用阅读或写作分数解释数学分数属于同期估计，不是考前预测；备考课程系数不能解释为培训的因果效果。',
    {'math score':'数学考试分数，原始分数范围 0 至 100。','reading score':'阅读考试分数，0 至 100。',
     'writing score':'写作考试分数，0 至 100。','gender':'原始性别分类。','race/ethnicity':'原始匿名群组 A 至 E；不得自行映射成实际族群。',
     'parental level of education':'父母受教育程度分类。','lunch':'午餐类型，standard 或 free/reduced；不能等同于精确家庭收入。',
     'test preparation course':'备考课程完成状态，completed 或 none；不是随机分组。'})
}

METHOD_DOCS = [
 ('method:rag','检索与计算','RAG 检索的是字段定义、数据边界和方法说明。所有统计量都在完整的筛选后数据上计算；不能把检索命中的几条文本当作统计样本。'),
 ('method:linear','线性回归','linear 只用于连续目标。数值特征在训练子集上中位数填补并标准化，分类变量在训练子集上填补并作独热编码，删去一个参考类别。系数描述条件线性关联，不是因果效果；本程序不报告显著性或 p 值。'),
 ('method:logistic','逻辑回归','logistic 只用于 Adult 的 income_gt_50k 二元目标，使用带 L2 正则的逻辑回归。输出类别 1 的概率。阈值固定 0.5；概率未额外校准。报告 ROC AUC、平均精确率、F1、准确率、平衡准确率、Brier 分数和对数损失。'),
 ('method:forest','随机森林','random_forest 对连续目标使用回归森林，对收入二元目标使用分类森林。固定 120 棵树、最大深度 12、叶节点至少 5 个样本、随机种子 42、一个并行工作线程。不支持现场超参数搜索。'),
 ('method:split','训练与交互验证','填补、标准化、独热编码仅在训练子集拟合。房价和学生随机划分用 80% 训练和 20% 交互验证；Adult 使用原始非目标字段组成的记录指纹分组，在分层分组五折中固定取首折验证，相同预测字段记录不跨组；NBA 按完整赛季时间划分，或按球员分组划分。反复试模型会对交互验证集过拟合，不能把它作为未使用的最终测试集。'),
 ('method:importance','特征重要性与解释','随机森林报告交互验证集上的置换重要性，3 次置换的均值及标准差，不是置信区间，也不是因果效应。相关特征之间可分摊重要性。线性系数的数值变量单位是训练样本的一个标准差。'),
 ('method:unsupported','能力边界','仅支持四个已注册数据集；筛选条件之间是 AND，不支持 OR、任意 SQL/Python、外部网页、神经网络、XGBoost、因果推断、显著性检验、自动调参和未观测的未来事件。必须明确拒绝范围外要求，不能降级后假装完成。'),
 ('method:units','相关和预测','皮尔逊相关只度量线性关联。回归报告 MAE、RMSE、R²；R² 允许为负，负值表示比用验证集均值作常数预测还差。基线模型仅从训练子集学习均值或正类比例。点预测不包含预测区间。')
]

class DataError(ValueError):
    pass

class Catalog:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def path(self, dataset: str) -> Path:
        if dataset not in SPECS:
            raise DataError('未注册的数据集。')
        return self.root / 'processed' / f'{dataset}.csv'

    def load(self, dataset: str):
        path = self.path(dataset)
        manifest_path = self.root / 'manifest.json'
        if not path.is_file() or not manifest_path.is_file():
            raise DataError(f'{SPECS[dataset].label}尚未准备。先运行数据准备程序；不会自动替换为合成数据。')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entry = manifest['datasets'][dataset]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise DataError('数据 SHA-256 与准备清单不一致；重新核实来源并准备数据。')
        category_types = {'MSSubClass': str} if dataset == 'house' else {c: str for c in ('draft_year','draft_round','draft_number')} if dataset == 'nba' else {}
        df = pd.read_csv(path, keep_default_na=False, na_values=[''], dtype=category_types, low_memory=False)
        # 字符串 None/NA 可能是类别，不在运行时擅自改为数值缺失。
        if df.columns.duplicated().any() or '_row_id' not in df:
            raise DataError('字段重复或缺少稳定行号。')
        if df['_row_id'].isna().any() or not df['_row_id'].is_unique:
            raise DataError('稳定行号不唯一。')
        if len(df) != entry['rows']:
            raise DataError('数据行数与清单不一致。')
        missing = set(SPECS[dataset].targets) - set(df.columns)
        if missing:
            raise DataError(f'缺少目标字段：{sorted(missing)}')
        return df, entry

    def available(self):
        records = []
        for name, spec in SPECS.items():
            path = self.path(name)
            records.append({'id':name,'label':spec.label,'present':path.is_file(),'source':spec.source})
        return records

    def schema(self, dataset):
        df, entry = self.load(dataset)
        rows=[]
        for c in df.columns:
            if c.startswith('_'): continue
            s=df[c]
            row={'column':c,'dtype':str(s.dtype),'missing':int(s.isna().sum()),'description':SPECS[dataset].descriptions.get(c,'以随原始数据提供的数据字典为准。')}
            if not pd.api.types.is_numeric_dtype(s):
                unique=s.dropna().astype(str).unique()
                row['examples']=sorted(unique)[:12]
                row['unique_count']=len(unique)
            rows.append(row)
        return {'id':dataset,'label':SPECS[dataset].label,'rows':len(df),'source_kind':entry['source_kind'],
                'targets':SPECS[dataset].targets,'default_features':[c for c in SPECS[dataset].defaults if c in df],
                'warning':SPECS[dataset].warning,'columns':rows,'sha256':entry['sha256'],
                'provenance_status':entry.get('provenance_status','unverified'),
                'source_note':entry.get('source_note','数据副本的真实性不由文件哈希证明。')}
