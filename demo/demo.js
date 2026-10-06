'use strict';
const $ = id => document.getElementById(id);
const STATS = {
  max: ['Maximum', 'Largest recorded value, not the largest absolute excursion.'],
  mean: ['Mean', 'Average signal level; captures any offset from zero.'],
  var: ['Variance', 'Spread around the mean, expressed on a squared scale.'],
  std: ['Standard deviation', 'Spread around the mean on the original signal scale.'],
  rms: ['RMS', 'Root mean square: overall signal magnitude, including its mean.'],
  kurtosis: ['Kurtosis', 'Shape statistic sensitive to extreme values and tails.'],
  skew: ['Skewness', 'Asymmetry of the recorded value distribution.'],
  ptp: ['Peak to peak', 'Difference between the highest and lowest recorded values.']
};
const SENSORS = {Motor:'Motor',B1_X:'Bearing 1 · X axis',B1_Y:'Bearing 1 · Y axis',B1_Z:'Bearing 1 · Z axis',B2_X:'Bearing 2 · X axis',B2_Y:'Bearing 2 · Y axis',B2_Z:'Bearing 2 · Z axis',Gearbox:'Gearbox',Tachometer:'Tachometer'};
function predict(tree, features) {
  let n = 0; const path = [];
  while (tree.left[n] !== -1) {
    const f = tree.feature[n], v = Math.fround(features[f]), threshold = tree.threshold[n], left = v <= threshold;
    path.push({f, v, threshold, left}); n = left ? tree.left[n] : tree.right[n];
  }
  return {category: tree.category[n], path};
}
function resultAssessment(data, example, prediction) {
  const labelMatches = prediction === example.category;
  return {text: labelMatches ? '✓ Correct for this example' : '× Wrong for this example',
    className: labelMatches ? 'match' : 'match mismatch', labelMatches, validatedDiagnosis: false};
}
function friendlyCondition(label) {
  if (label === 'No Fault') return 'No fault recorded';
  return label.split('&').map(part => {
    const lower=part.toLowerCase(), bearing=part.match(/Bearing \((\d)\)/);
    if (bearing) {
      const problem=lower.includes('inner')?'inner ring damage':lower.includes('outer')?'outer ring damage':lower.includes('ball')?'ball damage':'multiple faults';
      return `Bearing ${bearing[1]}: ${problem}`;
    }
    return lower.includes('coupling')?'Shaft bent near the joint':'Shaft bent in the middle';
  }).join(' + ');
}
function plainExplanation(label) {
  if(label==='No Fault')return 'The computer thinks the movement looks like the examples marked as having no problem. It cannot guarantee that a machine is safe.';
  const parts=[];
  if(label.includes('Bearing'))parts.push('A bearing is a small part that supports a spinning rod, rather like a wheel turning around its axle. This example describes damage to one of those support parts.');
  if(label.includes('Shaft'))parts.push('A shaft is a rod that spins inside the machine. A bent rod can wobble as it turns, rather like a bent bicycle wheel.');
  return parts.join(' ');
}
function format(value) { return Number(value).toLocaleString('en-US', {maximumSignificantDigits:5}); }
function node(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; }
function featureInfo(name) { const cut=name.indexOf('_'), stat=name.slice(0,cut), sensor=name.slice(cut+1); return {stat, sensor, title:`${SENSORS[sensor]} · ${STATS[stat][0]}`}; }
function sampleRange(mode, count) { return mode==='first' ? [0, Math.floor(count/4)-1] : mode==='middle' ? [Math.floor(count/4),Math.floor(count*3/4)-1] : mode==='last' ? [Math.floor(count*3/4),count-1] : [0,count-1]; }
function plot(example, reference, compare, mode, sampleRate) {
  const [start,end] = sampleRange(mode,example.samples);
  const selected=example.signal.filter(p=>p[0]>=start&&p[0]<=end);
  const baseline=reference.signal.filter(p=>p[0]>=start&&p[0]<=end);
  const values=(compare ? [...selected,...baseline] : selected).map(p=>p[1]);
  let lo=Math.min(...values),hi=Math.max(...values); const pad=(hi-lo)*.12||1; lo-=pad;hi+=pad;
  const x=i=>65+(i-start)/(end-start)*672,y=v=>239-(v-lo)/(hi-lo)*217;
  const g=$('chart-content');g.replaceChildren();
  const add=(tag,attrs,text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;g.append(e);return e;};
  for(let i=0;i<=4;i++) {const v=lo+(hi-lo)*i/4;add('line',{x1:65,x2:737,y1:y(v),y2:y(v),stroke:'#293849'});add('text',{x:56,y:y(v)+4,'text-anchor':'end',fill:'#95a9c2','font-size':10},Number(v.toPrecision(3)).toString());}
  for(let i=0;i<=4;i++){const sample=Math.round(start+(end-start)*i/4);add('text',{x:x(sample),y:257,'text-anchor':i===0?'start':i===4?'end':'middle',fill:'#95a9c2','font-size':10},(sample/sampleRate).toFixed(1));}
  add('text',{x:65,y:12,fill:'#95a9c2','font-size':10},'Motor acceleration (g)');add('text',{x:737,y:283,'text-anchor':'end',fill:'#95a9c2','font-size':10},'Time (seconds) →');
  if(compare)add('polyline',{points:baseline.map(([i,v])=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#d3a16c','stroke-width':1.4,'stroke-dasharray':'4 3',opacity:.7});
  add('polyline',{points:selected.map(([i,v])=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#aac9ff','stroke-width':1.4});
  $('reference-legend').hidden=!compare;
  $('chart-desc').textContent=`Motor channel for ${example.label}. ${compare?'Compared with the No Fault reference on a shared vertical scale. ':''}Samples ${start} to ${end}. Peak-preserving overview; not a full waveform.`;
  $('chart-summary').textContent=compare&&example.id===reference.id?'Both lines show the same no-problem example, so they sit on top of each other. Try “A damaged part” to compare two different recordings.':compare?'Blue is your chosen recording. Gold is an example marked as having no problem. The same scale makes them comparable.':'Showing the selected example only.';
  $('chart-summary').textContent+=` Recording: ${example.samples.toLocaleString('en-US')} samples · ${(example.samples/sampleRate).toFixed(2)} seconds.`;
}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=node('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function validateData(d){
  if(![25,50,75].includes(d.rpm)||d.kind!=='real'||!Array.isArray(d.examples)||d.examples.length!==39||!Array.isArray(d.features)||d.features.length!==72||!d.tree)throw Error('Invalid data');
  if(d.schemaVersion!==3||d.samplingRateHz!==6400||d.modelTrainingRpm!==d.rpm||!Number.isFinite(d.accuracy)||d.accuracy<0||d.accuracy>1)throw Error('Invalid provenance');
  const t=d.tree,n=t.left.length;
  if(!n||!['right','feature','threshold','category'].every(k=>Array.isArray(t[k])&&t[k].length===n))throw Error('Invalid tree');
  const seen=new Set(),active=new Set();
  function visit(i){if(!Number.isInteger(i)||i<0||i>=n||active.has(i))throw Error('Invalid tree edge');if(seen.has(i))return;active.add(i);if(t.left[i]!==-1){if(!Number.isInteger(t.feature[i])||t.feature[i]<0||t.feature[i]>=72||!Number.isFinite(t.threshold[i]))throw Error('Invalid split');visit(t.left[i]);visit(t.right[i]);}else if(t.right[i]!==-1||!d.categories[t.category[i]])throw Error('Invalid leaf');active.delete(i);seen.add(i);}visit(0);
  if(new Set(d.examples.map(e=>e.id)).size!==39||new Set(d.examples.map(e=>e.category)).size!==39)throw Error('Duplicate examples');
  for(const e of d.examples){if(e.kind!==d.kind||e.rpm!==d.rpm||d.categories[e.category]!==e.label||e.features.length!==72||!e.features.every(Number.isFinite)||!d.testIndices.includes(e.id)||d.trainIndices.includes(e.id)||!Number.isInteger(e.samples)||e.samples<4||!e.signal.length||!e.signal.every((p,i)=>p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[0]<e.samples&&(!i||p[0]>e.signal[i-1][0]))||predict(t,e.features).category!==e.expectedPrediction)throw Error('Invalid example');
    if(!/^[a-z0-9_]+\.csv$/.test(e.downloadName)|| e.labelBasis!=='source_filename')throw Error('Invalid download provenance');
  }
}
let requestVersion=0, handlers;
async function init(){
 const version=++requestVersion,rpm=$('dataset').value,url=rpm==='25'?'data.json?v=10':`real-${rpm}.json?v=10`;
 handlers?.abort();handlers=new AbortController();const on=(el,event,fn)=>el.addEventListener(event,fn,{signal:handlers.signal});
 $('demo').hidden=true;$('status').textContent='Loading and checking selected dataset…';
 try{
  const response=await fetch(url);if(!response.ok)throw Error('Data unavailable');const data=await response.json();if(version!==requestVersion)return;validateData(data);if(data.rpm!==Number(rpm))throw Error('Wrong speed response');
  $('dataset-note').textContent=`Original laboratory recordings with filename-derived labels. This model was trained only at ${data.modelTrainingRpm} RPM; displayed examples were held out before training.`;
  $('data-caption').textContent=`Real ${data.rpm} RPM recordings from David Jensen's Figshare dataset (CC BY 4.0). Filename-derived labels and row-level holdout; no maintenance advice.`;
  $('evaluation-scope').textContent=`The model learned from ${data.trainCount} real ${data.rpm} RPM recordings and was tested on ${data.testCount} held-out recordings. The 39 displayed examples come from that test.`;
  $('download-dataset').href=url;
  $('reproduction-command').textContent=`python fault_diagnosis.py evaluate --rpm ${data.rpm} --domain time --model tree`;
  const reference=data.examples.find(e=>e.label==='No Fault'),select=$('measurement');
  select.replaceChildren();for(const e of data.examples){const option=node('option',`${friendlyCondition(e.label)} · ${e.downloadName.replace('.csv','_features.csv')}`);option.value=e.id;select.append(option);}
  const sensor=$('sensor');sensor.replaceChildren();for(const [id,label] of Object.entries(SENSORS)){const option=node('option',label);option.value=id;sensor.append(option);}
  select.value=String(reference.id);sensor.value='Motor';let analysis=null;
  const current=()=>data.examples.find(e=>String(e.id)===select.value);
  const feature=(e,name)=>e.features[data.features.indexOf(name)];
  const draw=()=>plot(current(),reference,$('compare').checked,$('window').value,data.samplingRateHz);
  function table(){const e=current(),used=new Set(analysis?analysis.path.map(p=>p.f):[]);$('feature-rows').replaceChildren();for(const [key,[label,description]]of Object.entries(STATS)){const name=`${key}_${sensor.value}`,index=data.features.indexOf(name),row=node('tr');if(used.has(index))row.className='used';const title=node('td',label);if(used.has(index))title.append(node('span','Used in decision','used-tag'));row.append(title,node('td',format(e.features[index])),node('td',format(reference.features[index])),node('td',description));$('feature-rows').append(row);}$('sensor-description').textContent=`${SENSORS[sensor.value]} · 8 statistics from the full recording. The model receives values from all 9 channels, regardless of which channel you explore here.`;}
  function update(){analysis=null;const e=current();$('selected-file').textContent=`Real ${data.rpm} RPM · ${e.source} · Feature export: ${e.downloadName.replace('.csv','_features.csv')}`;$('download-raw').hidden=false;$('download-raw').href=data.archiveUrl;$('download-raw').textContent=`Download original ${data.rpm} RPM archive from Figshare (~5 GB) ↓`;draw();table();document.querySelectorAll('.machine-part').forEach(part=>part.classList.remove('highlight'));if(e.label.includes('Bearing (1)'))$('part-b1').classList.add('highlight');if(e.label.includes('Bearing (2)'))$('part-b2').classList.add('highlight');if(e.label.includes('Shaft'))$('part-shaft').classList.add('highlight');$('machine-caption').textContent=e.label==='No Fault'?'No part is marked as faulty in this example.':'Highlighted in gold: the part described as faulty in this example.';$('selected-label').textContent=`The recorded description: ${friendlyCondition(e.label)}`;$('example-guide').textContent=e.label==='No Fault'?'This recording was marked as having no problem. Use it as a starting point, then try the damaged-part example.':plainExplanation(e.label);$('result').replaceChildren(node('h4','Try it and find out.'),node('p','Press “Check this example”. You do not need to enter any numbers or understand the graph.'));$('download-result').disabled=true;$('decision-panel').hidden=true;$('decision-panel').open=false;$('decision-path').replaceChildren();$('signal-metrics').replaceChildren();for(const [stat,label]of [['rms','Motor · RMS'],['ptp','Motor · peak to peak'],['std','Motor · standard deviation']]){const metric=node('div');metric.append(node('span',label),node('strong',format(feature(e,`${stat}_Motor`))));$('signal-metrics').append(metric);}$('source').textContent=`Original source filename: ${e.source} · Feature-table row ${e.id} (zero-based) · Recorded category ${e.category}: ${e.label}.`;$('hash').textContent=`Original measurement SHA-256: ${e.sha256}`;$('analyze').textContent='Check this example →';document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed','false'));}
  on(select,'change',update);on($('compare'),'change',draw);on($('window'),'change',draw);on(sensor,'change',table);
  const cases={healthy:reference,bearing:data.examples.find(e=>e.category===11),shaft:data.examples.find(e=>e.category===38),disagreement:data.examples.find(e=>e.category!==e.expectedPrediction)};
  document.querySelectorAll('[data-case]').forEach(button=>{button.disabled=!cases[button.dataset.case];on(button,'click',()=>{select.value=String(cases[button.dataset.case].id);update();button.setAttribute('aria-pressed','true');});});
  on($('analyze'),'click',()=>{const e=current();analysis=predict(data.tree,e.features);const assessment=resultAssessment(data,e,analysis.category),match=assessment.labelMatches,result=$('result');result.replaceChildren(node('p',`THE COMPUTER’S ANSWER · ${data.rpm} RPM MODEL`,'eyebrow'),node('h4',friendlyCondition(data.categories[analysis.category])),node('span',assessment.text,assessment.className),node('p',`Recorded answer: ${friendlyCondition(e.label)}`),node('p',plainExplanation(data.categories[analysis.category]).replace('This example describes damage','That answer means possible damage'),'small'),node('p',match?'The prediction and the original recording label agree. This is one held-out example; it does not establish reliability on a new machine.':'The computer chose a different problem from the one described in the original recording. This is a model error relative to that label, not evidence that the original label is wrong.'));$('analyze').textContent='Check again →';$('download-result').disabled=false;$('decision-panel').hidden=false;$('path-count').textContent=`${analysis.path.length} steps`;$('decision-path').replaceChildren();analysis.path.forEach(step=>{const info=featureInfo(data.features[step.f]),item=node('li');item.append(node('strong',info.title),node('code',`${format(step.v)} ${step.left?'≤':'>'} ${format(step.threshold)}`),node('small',`${step.left?'Left':'Right'} branch · ${STATS[info.stat][1]}`));$('decision-path').append(item);});table();});
  on($('download-result'),'click',()=>{if(!analysis)return;const e=current();download(e.downloadName.replace('.csv','_analysis.json'),JSON.stringify({dataKind:data.kind,rpm:data.rpm,modelTrainingRpm:data.modelTrainingRpm,labelBasis:e.labelBasis,sourceDataset:data.sourceDataset,doi:data.doi,license:data.license,source:e.source,row:e.id,recordedLabel:e.label,predictedLabel:data.categories[analysis.category],matchesRecordedLabel:analysis.category===e.category,interpretation:'measured_same_speed_row_holdout',independentRunValidated:false,signalSha256:e.sha256,featureTableSha256:data.featureTableSha256,model:`${data.rpm} RPM time-domain decision tree`,seed:data.seed,holdoutAccuracy:data.accuracy,path:analysis.path.map(p=>({...p,feature:data.features[p.f]})),limitations:'Academic row-level evaluation; related acquisitions may cross the split. No independent-machine validation, calibrated confidence or maintenance recommendation.'},null,2),'application/json');});
  on($('download-features'),'click',()=>{const e=current(),csv=[['data_kind','rpm','feature','source_row','selected_value','reference_source_row','reference_value'],...data.features.map((f,i)=>[data.kind,data.rpm,f,e.id,e.features[i],reference.id,reference.features[i]])].map(row=>row.join(',')).join('\n');download(e.downloadName.replace('.csv','_features.csv'),csv,'text/csv;charset=utf-8');});
  $('accuracy-caption').textContent=`correct in the real ${data.rpm} RPM test`;$('accuracy').textContent=`${(data.accuracy*100).toFixed(1)}%`;$('evaluation').textContent=`It correctly labelled ${Math.round(data.accuracy*data.testCount)} of ${data.testCount} test recordings, across ${Object.keys(data.categories).length} conditions. This is not confidence in an individual answer.`;
  const correct=data.examples.filter(e=>e.category===predict(data.tree,e.features).category).length;$('example-score').textContent=`The 39 examples on this page include ${correct} correct answers and ${data.examples.length-correct} mistakes relative to their recorded labels. They are only part of the full test.`;
  $('table-hash').textContent=`Real ${data.rpm} RPM training feature-table SHA-256: ${data.featureTableSha256}`;$('versions').textContent=`Export environment: scikit-learn ${data.versions.sklearn}, NumPy ${data.versions.numpy}. Split seed ${data.seed}. Numbers in the decision path are rounded for display; downloaded thresholds retain full precision.`;
  update();$('status').textContent='';$('demo').hidden=false;$('retry').hidden=true;
 }catch(error){if(version!==requestVersion)return;$('demo').hidden=true;$('status').textContent='The measurements could not be loaded or verified. Please try again. The project and methodology links remain available below.';$('retry').hidden=false;}
}
if(typeof document!=='undefined'){$('retry').addEventListener('click',init);$('dataset').addEventListener('change',init);init();}
if(typeof module!=='undefined')module.exports={predict,sampleRange,validateData,featureInfo,friendlyCondition,resultAssessment};
