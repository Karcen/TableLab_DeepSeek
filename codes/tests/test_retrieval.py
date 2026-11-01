import pytest
from tablelab.retrieval import Retriever

@pytest.mark.parametrize('dataset,question,expected',[
 ('house','地上居住面积平方英尺','house:field:GrLivArea'),
 ('house','房屋成交价美元','house:field:SalePrice'),
 ('adult','是否超过5万美元收入类别 income_gt_50k','adult:field:income_gt_50k'),
 ('adult','每周工作时长小时','adult:field:hours-per-week'),
 ('nba','赛季场均得分 pts','nba:field:pts'),
 ('nba','球员身高厘米','nba:field:player_height'),
 ('students','备考课程完成状态','students:field:test preparation course'),
 ('students','数学考试分数','students:field:math score'),
 ('house','随机森林置换重要性','method:importance'),
 ('adult','训练验证数据泄漏','method:split')])
def test_retrieval_expected_document_in_top6(catalog,dataset,question,expected):
    result=Retriever(catalog,dataset).search(question)
    assert expected in [r['id'] for r in result]
    assert all(0<=r['similarity']<=1 for r in result)
    assert all(r['id'].startswith(dataset+':') or r['id'].startswith('method:') for r in result)
