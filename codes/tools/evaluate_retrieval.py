"""只评价字段/方法检索，不评价 LLM 问答正确率。固定题集不代表未知问题分布。"""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog
from tablelab.retrieval import Retriever
from tablelab.engine import write_json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CASES=[
 ('house','房屋成交价 SalePrice 的单位是什么？','house:field:SalePrice'),
 ('house','地上居住面积 GrLivArea 是否包含地下室？','house:field:GrLivArea'),
 ('house','房屋所在社区 Neighborhood 是分类变量吗？','house:field:Neighborhood'),
 ('house','线性回归系数能解释成因果效果吗？','method:linear'),
 ('adult','高收入 income_gt_50k 中的 1 是什么意思？','adult:field:income_gt_50k'),
 ('adult','原始收入 income 是否记录了具体金额？','adult:field:income'),
 ('adult','fnlwgt 是否是样本权重？','adult:field:fnlwgt'),
 ('adult','education-num 是否直接等于受教育年数？','adult:field:education-num'),
 ('nba','player_height 的单位是什么？','nba:field:player_height'),
 ('nba','场均得分 pts 是什么？','nba:field:pts'),
 ('nba','season 赛季如何用于时间划分？','nba:field:season'),
 ('nba','player_name 是唯一球员身份吗？','nba:field:player_name'),
 ('students','math score 数学分数的范围是什么？','students:field:math score'),
 ('students','lunch 午餐类型是否表示精确家庭收入？','students:field:lunch'),
 ('students','test preparation course 课程是随机分组吗？','students:field:test preparation course'),
 ('students','race/ethnicity 群组 A 到 E 对应哪些族群？','students:field:race/ethnicity')]

def main():
    write_json(ROOT/'verification/retrieval_protocol.json',{'k':6,'cases':CASES,'scope':'人工固定字段释义题；不是现场未知问题样本'})
    cat=Catalog(ROOT/'data');rows=[]
    for dataset,q,gold in CASES:
        docs=Retriever(cat,dataset).search(q,k=6);ids=[d['id'] for d in docs]
        rank=ids.index(gold)+1 if gold in ids else None
        rows.append({'dataset':dataset,'question':q,'expected_document':gold,'rank':rank,'hit_at_6':rank is not None,'reciprocal_rank_at_6':1/rank if rank is not None else 0,'retrieved_ids':json.dumps(ids,ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(ROOT/'tables/retrieval_checks.csv',index=False,encoding='utf-8-sig')
    write_json(ROOT/'verification/retrieval_checks.json',{'n':len(rows),'hits_at_6':sum(r['hit_at_6'] for r in rows),'mrr_at_6':sum(r['reciprocal_rank_at_6'] for r in rows)/len(rows),'llm_evaluated':False,'scope':'固定人工字段释义题，多数含原字段名；只能证明这些检索案例，不代表任意中文问题','cases':rows})
    print('hits',sum(r['hit_at_6'] for r in rows),'/',len(rows))
if __name__=='__main__':main()
