// node demo/check_browser_model.cjs /path/to/exported/data.json
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const ctx={module:{exports:{}}};vm.createContext(ctx);vm.runInContext(fs.readFileSync(__dirname+'/demo.js','utf8'),ctx);
const {predict,sampleRange,validateData,featureInfo,friendlyCondition,scenarioSummary,resultAssessment}=ctx.module.exports;
validateData(data);
assert.equal(new Set(data.examples.map(e=>e.category)).size,39);
assert.equal(new Set(data.examples.map(e=>friendlyCondition(e.label))).size,39);
assert.equal(friendlyCondition('No Fault'),'No fault recorded');
assert.equal(friendlyCondition('Bearing (1) Fault (outer race)'),'Bearing 1: outer ring damage');
assert.equal(friendlyCondition('Shaft Fault (Coupling end bent)'),'Shaft bent near the joint');
assert.equal(new Set(data.trainIndices).size,data.trainCount);
assert.equal(new Set(data.testIndices).size,data.testCount);
assert(data.testIndices.every(i=>!data.trainIndices.includes(i)));
for(const e of data.examples){
 assert.equal(predict(data.tree,e.features).category,e.expectedPrediction);
 assert.equal(e.samples,data.kind==='real'?64000:4096);assert(e.signal.length<=480);
 for(const mode of ['full','first','middle','last']){const [a,b]=sampleRange(mode,e.samples);assert(a>=0&&b<e.samples&&a<b);assert(e.signal.filter(p=>p[0]>=a&&p[0]<=b).length>1);}
 for(const f of data.features)assert(featureInfo(f).title);
}
if(data.kind==='real')assert.equal(data.examples.filter(e=>predict(data.tree,e.features).category!==e.category).length,3);
else assert.equal(data.accuracy,null);
const cyclic=JSON.parse(JSON.stringify(data));cyclic.tree.left[0]=0;assert.throws(()=>validateData(cyclic));
const disguised=JSON.parse(JSON.stringify(data));disguised.examples[0].kind='unknown';assert.throws(()=>validateData(disguised));
const tree={left:[1,-1,-1],right:[2,-1,-1],feature:[0,-2,-2],threshold:[1,-2,-2],category:[0,10,20]};
assert.equal(predict(tree,[1]).category,10);assert.equal(predict(tree,[1+1e-8]).category,10);assert.equal(predict(tree,[1.1]).category,20);
const bad=JSON.parse(JSON.stringify(data));bad.examples[0].features[0]=null;assert.throws(()=>validateData(bad));
const leaked=JSON.parse(JSON.stringify(data));leaked.trainIndices.push(leaked.examples[0].id);assert.throws(()=>validateData(leaked));
console.log(`PASS: ${data.rpm} RPM ${data.kind}: 39 predictions, split separation, 156 chart ranges, feature names, float32 boundaries, cycle and provenance rejection.`);

const summary=scenarioSummary(data);
assert.equal(summary.total,39);
assert.equal(summary.labelMatches,{25:36,50:5,75:6}[data.rpm]);
assert.equal(summary.unchanged,{25:39,50:4,75:6}[data.rpm]);
// A correct source-label guess can still be a changed prediction, and an
// unchanged prediction can still disagree with the inherited label.
for(const [category,parent,predicted,labelMatch,unchanged] of [
 [1,2,1,true,false],[1,2,2,false,true],[1,1,1,true,true],[1,1,2,false,false]
]){
 const a=resultAssessment({kind:'synthetic'},{category,parentPrediction:parent},predicted);
 assert.equal(a.labelMatches,labelMatch);assert.equal(a.sameAsParent,unchanged);
 assert.equal(a.validatedDiagnosis,false);assert.equal(a.className,'match sensitivity');
 assert.equal(a.text,unchanged?'Prediction unchanged by transformation':'Prediction changed by transformation');
}
assert.equal(resultAssessment({kind:'real'},{category:1,parentPrediction:1},2).text,'× Wrong for this example');
console.log('PASS: source-label agreement and parent-prediction stability remain distinct; synthetic outputs never claim validated diagnosis.');
