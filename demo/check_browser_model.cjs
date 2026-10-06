// node demo/check_browser_model.cjs /path/to/exported/data.json
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const ctx={module:{exports:{}}};vm.createContext(ctx);vm.runInContext(fs.readFileSync(__dirname+'/demo.js','utf8'),ctx);
const {predict,sampleRange,validateData,featureInfo,friendlyCondition,resultAssessment}=ctx.module.exports;
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
 assert.equal(e.samples,e.source==='Bearing (2) Ball 50 Trial 17.csv'?34044:64000);assert(e.signal.length<=480);
 for(const mode of ['full','first','middle','last']){const [a,b]=sampleRange(mode,e.samples);assert(a>=0&&b<e.samples&&a<b);assert(e.signal.filter(p=>p[0]>=a&&p[0]<=b).length>1);}
 for(const f of data.features)assert(featureInfo(f).title);
}
assert.equal(data.kind,'real');assert.equal(data.modelTrainingRpm,data.rpm);
assert.equal(data.examples.filter(e=>e.labelBasis==='source_filename').length,39);
const cyclic=JSON.parse(JSON.stringify(data));cyclic.tree.left[0]=0;assert.throws(()=>validateData(cyclic));
const disguised=JSON.parse(JSON.stringify(data));disguised.examples[0].kind='unknown';assert.throws(()=>validateData(disguised));
const tree={left:[1,-1,-1],right:[2,-1,-1],feature:[0,-2,-2],threshold:[1,-2,-2],category:[0,10,20]};
assert.equal(predict(tree,[1]).category,10);assert.equal(predict(tree,[1+1e-8]).category,10);assert.equal(predict(tree,[1.1]).category,20);
const bad=JSON.parse(JSON.stringify(data));bad.examples[0].features[0]=null;assert.throws(()=>validateData(bad));
const leaked=JSON.parse(JSON.stringify(data));leaked.trainIndices.push(leaked.examples[0].id);assert.throws(()=>validateData(leaked));
console.log(`PASS: ${data.rpm} RPM ${data.kind}: 39 predictions, split separation, 156 chart ranges, feature names, float32 boundaries, cycle and provenance rejection.`);

for(const rpm of [25,50,75]) {
 const badSpeed=JSON.parse(JSON.stringify(data));badSpeed.modelTrainingRpm=rpm===25?50:25;badSpeed.rpm=rpm;
 assert.throws(()=>validateData(badSpeed));
}
const inherited=JSON.parse(JSON.stringify(data));inherited.examples[0].labelBasis='inherited_from_25rpm_parent';assert.throws(()=>validateData(inherited));
assert.equal(resultAssessment(data,{category:1},2).text,'× Wrong for this example');
console.log('PASS: same-speed models and source-filename provenance enforced; incorrect labels remain visible.');
