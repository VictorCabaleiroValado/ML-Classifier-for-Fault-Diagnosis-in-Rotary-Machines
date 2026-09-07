"""Export a browser decision tree and one deterministic held-out example per category."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fault_diagnosis as fd
raw_dir=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
path=fd.feature_path(25,'time'); df,X,y=fd.load_features(path,'time')
train,test=fd.split_indices(y)
model=fd.build_model('tree'); model.fit(X.iloc[train],y.iloc[train].fault_category)
t=model.tree_
tree={'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'category':[int(model.classes_[np.argmax(v)]) for v in t.value]}
def predict(x):
 n=0
 while tree['left'][n]!=-1:n=tree['left'][n] if np.float32(x[tree['feature'][n]])<=tree['threshold'][n] else tree['right'][n]
 return tree['category'][n]
pred=model.predict(X.iloc[test]); assert [predict(x) for x in X.iloc[test].to_numpy()]==pred.tolist()
categories=json.loads((fd.ROOT/'data/manifests/25_rpm_categories.json').read_text())
examples=[]
for c in sorted(y.fault_category.unique()):
 i=min(int(i) for i in test if y.iloc[i].fault_category==c)
 source=raw_dir/df.iloc[i].source_file
 samples=pd.read_csv(source,usecols=list(range(1,18,2)),skiprows=range(1,4),dtype=float); samples.columns=fd.SENSORS
 features=fd.signal_features(samples,'time')
 np.testing.assert_allclose(list(features.values()),X.iloc[i].values,rtol=1e-8,atol=1e-10)
 signal=samples.Motor.to_numpy(); envelope=[]
 for block in np.array_split(np.arange(len(signal)),240):
  ids=sorted({int(block[np.argmin(signal[block])]),int(block[np.argmax(signal[block])])})
  envelope.extend([[j,round(float(signal[j]),6)] for j in ids])
 examples.append({'id':i,'label':categories[str(c)],'category':int(c),'source':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'features':X.iloc[i].tolist(),'signal':envelope,'samples':len(signal),'expectedPrediction':predict(X.iloc[i].values)})
payload={'tree':tree,'categories':categories,'examples':examples,'features':list(X.columns),'trainCount':len(train),'testCount':len(test),'accuracy':float(np.mean(pred==y.iloc[test].fault_category)),'seed':42,'trainIndices':train.tolist(),'testIndices':test.tolist(),'featureTableSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'versions':{'sklearn':fd.sklearn.__version__,'numpy':np.__version__}}
(out/'data.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))
print(f'Exported {len(examples)} held-out examples; holdout accuracy {payload["accuracy"]:.3f}; parity verified on all {len(test)} test rows')
