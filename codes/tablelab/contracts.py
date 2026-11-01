"""语言模型输出是数据，不是可执行程序。边界校验失败即停止。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

DatasetId = Literal['house', 'adult', 'nba', 'students']
Scalar = str | int | float | bool

class BoundaryModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Filter(BoundaryModel):
    column: str = Field(min_length=1, max_length=80)
    op: Literal['eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'is_missing', 'not_missing']
    value: Scalar | list[Scalar] | None = None

    @model_validator(mode='after')
    def check_value(self):
        if self.op in ('is_missing', 'not_missing'):
            if self.value is not None:
                raise ValueError('缺失值筛选不接受 value。')
        elif self.op == 'in':
            if not isinstance(self.value, list) or not self.value or len(self.value) > 30:
                raise ValueError('in 筛选必须给出 1 到 30 个值。')
        elif self.value is None or isinstance(self.value, list):
            raise ValueError('此筛选操作需要单个非空值。')
        return self

class Plan(BoundaryModel):
    dataset: DatasetId
    operation: Literal['explain', 'describe', 'aggregate', 'correlation', 'plot', 'train', 'predict', 'unsupported', 'clarify']
    columns: list[str] = Field(default_factory=list, max_length=30)
    filters: list[Filter] = Field(default_factory=list, max_length=8)
    group_by: list[str] = Field(default_factory=list, max_length=2)
    metric: Literal['count', 'mean', 'median', 'sum', 'min', 'max', 'std'] = 'mean'
    target: str | None = None
    features: list[str] = Field(default_factory=list, max_length=30)
    algorithm: Literal['linear', 'logistic', 'random_forest'] = 'linear'
    split: Literal['random', 'time', 'group'] = 'random'
    context: Literal['retrospective', 'prospective'] = 'retrospective'
    chart: Literal['histogram', 'scatter', 'bar'] = 'histogram'
    limit: int = Field(default=20, ge=1, le=100)
    descending: bool = True
    model_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    record: dict[str, Scalar | None] = Field(default_factory=dict, max_length=30)
    message: str = Field(default='', max_length=1000)

    @model_validator(mode='after')
    def structure(self):
        for values in (self.columns, self.features, self.group_by):
            if len(values) != len(set(values)):
                raise ValueError('字段列表不能重复。')
        if self.operation in ('unsupported', 'clarify') and not self.message.strip():
            raise ValueError('不支持或待澄清方案必须说明原因。')
        if self.operation in ('train', 'predict') and self.columns:
            raise ValueError('建模使用 features，单条预测使用 record，均不接受 columns。')
        if self.operation == 'aggregate' and self.metric == 'count' and len(self.columns)>1:
            raise ValueError('计数最多指定一个字段；不指定字段时计算行数。')
        if self.operation == 'train':
            if not self.target or not self.features:
                raise ValueError('建模必须明确目标变量和至少一个自变量。')
            if self.target in self.features:
                raise ValueError('目标变量不能同时作为自变量。')
        if self.operation == 'predict' and (not self.model_id or not self.record):
            raise ValueError('新样本预测必须提供当前会话 model_id 和特征值。')
        if self.operation == 'correlation' and len(self.columns) != 2:
            raise ValueError('相关分析必须且只能指定两个数值字段。')
        if self.operation == 'plot':
            expected = 2 if self.chart == 'scatter' else 0 if self.chart == 'bar' and self.metric == 'count' and not self.columns else 1
            if len(self.columns) != expected:
                raise ValueError(f'{self.chart} 需要 {expected} 个数值字段。')
            if self.chart == 'bar' and not self.group_by:
                raise ValueError('柱形图必须指定分组字段。')
        if self.operation == 'aggregate' and self.metric != 'count' and len(self.columns) != 1:
            raise ValueError('除计数外，聚合操作须且只能指定一个数值字段。')
        return self

class Question(BoundaryModel):
    dataset: DatasetId
    question: str = Field(min_length=1, max_length=1500)
    previous: Plan | None = None
    model_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')

class ExecutionRequest(BoundaryModel):
    plan: Plan
    question: str = Field(default='', max_length=1500)
    evidence: list[dict] = Field(default_factory=list, max_length=8)
    proposal_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
