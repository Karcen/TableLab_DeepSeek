"""本地中英文字段说明稀疏检索；不把相似度称为答案置信度。"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .catalog import SPECS, METHOD_DOCS

class Retriever:
    def __init__(self, catalog, dataset):
        schema=catalog.schema(dataset)
        spec=SPECS[dataset]
        self.docs=[{'id':f'{dataset}:scope','title':spec.label+'数据边界','text':spec.warning,'source':spec.source}]
        for c in schema['columns']:
            self.docs.append({'id':f'{dataset}:field:{c["column"]}','title':c['column'],
                'text':c['column']+' '+c['description']+' 数据类型 '+c['dtype']+' 示例值 '+', '.join(c.get('examples',[])),
                'source':spec.source})
        self.docs += [{'id':i,'title':t,'text':body,'source':'本项目方法说明'} for i,t,body in METHOD_DOCS]
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=True,sublinear_tf=True)
        self.matrix=self.vectorizer.fit_transform(d['title']+' '+d['text'] for d in self.docs)

    def search(self, question: str, k: int = 6):
        if not question.strip(): raise ValueError('检索问题不能为空。')
        q=self.vectorizer.transform([question])
        scores=(self.matrix @ q.T).toarray().ravel()
        ids=np.argsort(-scores,kind='stable')[:k]
        return [{**self.docs[i],'similarity':round(float(scores[i]),6)} for i in ids if scores[i]>0]
