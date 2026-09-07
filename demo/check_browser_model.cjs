// Run with: node demo/check_browser_model.cjs /path/to/exported/data.json
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const code=fs.readFileSync(__dirname+'/demo.js','utf8').split("fetch('data.json')")[0];
const ctx={};vm.createContext(ctx);vm.runInContext(code,ctx);
assert.equal(data.examples.length,39);
for(const e of data.examples){assert(!data.trainIndices.includes(e.id));assert(data.testIndices.includes(e.id));assert.equal(ctx.predict(data.tree,e.features).category,e.expectedPrediction);assert.equal(e.features.length,72);assert(e.signal.length<=480);}
assert.equal(new Set(data.examples.map(e=>e.category)).size,39);
console.log('All 39 browser predictions match Python; holdout membership and signal dimensions verified.');
