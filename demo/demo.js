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
  if(label==='No Fault')return 'Its pattern resembles examples labelled as having no fault. This is not a guarantee that a machine is safe or healthy.';
  const parts=[];
  if(label.includes('Bearing'))parts.push('Bearings are parts that support a spinning rod and help it turn smoothly.');
  if(label.includes('Shaft'))parts.push('The shaft is the rod that spins; a bend can change its vibration pattern.');
  return parts.join(' ');
}
function format(value) { return Number(value).toLocaleString('en-US', {maximumSignificantDigits:5}); }
function node(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; }
function featureInfo(name) { const cut=name.indexOf('_'), stat=name.slice(0,cut), sensor=name.slice(cut+1); return {stat, sensor, title:`${SENSORS[sensor]} · ${STATS[stat][0]}`}; }
function sampleRange(mode, count) { return mode==='first' ? [0, Math.floor(count/4)-1] : mode==='middle' ? [Math.floor(count/4),Math.floor(count*3/4)-1] : mode==='last' ? [Math.floor(count*3/4),count-1] : [0,count-1]; }
function plot(example, reference, compare, mode) {
  const [start,end] = sampleRange(mode,example.samples);
  const selected=example.signal.filter(p=>p[0]>=start&&p[0]<=end);
  const baseline=reference.signal.filter(p=>p[0]>=start&&p[0]<=end);
  const values=(compare ? [...selected,...baseline] : selected).map(p=>p[1]);
  let lo=Math.min(...values),hi=Math.max(...values); const pad=(hi-lo)*.12||1; lo-=pad;hi+=pad;
  const x=i=>65+(i-start)/(end-start)*672,y=v=>239-(v-lo)/(hi-lo)*217;
  const g=$('chart-content');g.replaceChildren();
  const add=(tag,attrs,text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;g.append(e);return e;};
  for(let i=0;i<=4;i++) {const v=lo+(hi-lo)*i/4;add('line',{x1:65,x2:737,y1:y(v),y2:y(v),stroke:'#293849'});add('text',{x:56,y:y(v)+4,'text-anchor':'end',fill:'#95a9c2','font-size':10},Number(v.toPrecision(3)).toString());}
  for(let i=0;i<=4;i++){const sample=Math.round(start+(end-start)*i/4);add('text',{x:x(sample),y:257,'text-anchor':i===0?'start':i===4?'end':'middle',fill:'#95a9c2','font-size':10},sample.toLocaleString('en-US'));}
  add('text',{x:65,y:12,fill:'#95a9c2','font-size':10},'Vibration reading');add('text',{x:737,y:283,'text-anchor':'end',fill:'#95a9c2','font-size':10},'Reading number →');
  if(compare)add('polyline',{points:baseline.map(([i,v])=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#d3a16c','stroke-width':1.4,'stroke-dasharray':'4 3',opacity:.7});
  add('polyline',{points:selected.map(([i,v])=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#aac9ff','stroke-width':1.4});
  $('reference-legend').hidden=!compare;
  $('chart-desc').textContent=`Motor channel for ${example.label}. ${compare?'Compared with the No Fault reference on a shared vertical scale. ':''}Samples ${start} to ${end}. Peak-preserving overview; not a full waveform.`;
  $('chart-summary').textContent=compare&&example.id===reference.id?'You selected the no-fault reference, so the two lines overlap.':compare?'Both lines use the same scale so you can compare their patterns.':'Showing the selected example only.';
}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=node('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function validateData(d){
  if(!Array.isArray(d.examples)||d.examples.length!==39||!Array.isArray(d.features)||d.features.length!==72||!d.tree||!Number.isFinite(d.accuracy)||d.accuracy<0||d.accuracy>1)throw Error('Invalid data');
  for(const e of d.examples){if(e.features.length!==72||!e.features.every(Number.isFinite)||!d.testIndices.includes(e.id)||d.trainIndices.includes(e.id)||!e.signal.length||!e.signal.every(p=>p.length===2&&p.every(Number.isFinite))||predict(d.tree,e.features).category!==e.expectedPrediction)throw Error('Invalid example');}
}
async function init(){
 try{
  const response=await fetch('data.json');if(!response.ok)throw Error('Data unavailable');const data=await response.json();validateData(data);
  const reference=data.examples.find(e=>e.label==='No Fault'),select=$('measurement');
  select.replaceChildren();for(const e of data.examples){const option=node('option',friendlyCondition(e.label));option.value=e.id;select.append(option);}
  const sensor=$('sensor');sensor.replaceChildren();for(const [id,label] of Object.entries(SENSORS)){const option=node('option',label);option.value=id;sensor.append(option);}
  select.value=String(data.examples.find(e=>e.category===11).id);sensor.value='Motor';let analysis=null;
  const current=()=>data.examples.find(e=>String(e.id)===select.value);
  const feature=(e,name)=>e.features[data.features.indexOf(name)];
  const draw=()=>plot(current(),reference,$('compare').checked,$('window').value);
  function table(){const e=current(),used=new Set(analysis?analysis.path.map(p=>p.f):[]);$('feature-rows').replaceChildren();for(const [key,[label,description]]of Object.entries(STATS)){const name=`${key}_${sensor.value}`,index=data.features.indexOf(name),row=node('tr');if(used.has(index))row.className='used';const title=node('td',label);if(used.has(index))title.append(node('span','Used in decision','used-tag'));row.append(title,node('td',format(e.features[index])),node('td',format(reference.features[index])),node('td',description));$('feature-rows').append(row);}$('sensor-description').textContent=`${SENSORS[sensor.value]} · 8 statistics from the full recording. The model receives values from all 9 channels, regardless of which channel you explore here.`;}
  function update(){analysis=null;const e=current();draw();table();$('selected-label').textContent=`Recorded in this example: ${friendlyCondition(e.label)}`;$('result').replaceChildren(node('h4','What will it find?'),node('p','Select the button above to see the answer. The computer uses the full recording, not just the picture.'));$('download-result').disabled=true;$('decision-panel').hidden=true;$('decision-panel').open=false;$('decision-path').replaceChildren();$('signal-metrics').replaceChildren();for(const [stat,label]of [['rms','Motor · RMS'],['ptp','Motor · peak to peak'],['std','Motor · standard deviation']]){const metric=node('div');metric.append(node('span',label),node('strong',format(feature(e,`${stat}_Motor`))));$('signal-metrics').append(metric);}$('source').textContent=`Original source filename: ${e.source} · Feature-table row ${e.id} (zero-based) · Recorded category ${e.category}: ${e.label}.`;$('hash').textContent=`Measurement SHA-256: ${e.sha256}`;$('analyze').textContent='Show the prediction →';document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed','false'));}
  select.addEventListener('change',update);$('compare').addEventListener('change',draw);$('window').addEventListener('change',draw);sensor.addEventListener('change',table);
  const cases={healthy:reference,bearing:data.examples.find(e=>e.category===11),shaft:data.examples.find(e=>e.category===38),disagreement:data.examples.find(e=>e.category!==e.expectedPrediction)};
  document.querySelectorAll('[data-case]').forEach(button=>button.addEventListener('click',()=>{select.value=String(cases[button.dataset.case].id);update();button.setAttribute('aria-pressed','true');}));
  $('analyze').addEventListener('click',()=>{const e=current();analysis=predict(data.tree,e.features);const match=analysis.category===e.category,result=$('result');result.replaceChildren(node('p','THE COMPUTER’S ANSWER','eyebrow'),node('h4',friendlyCondition(data.categories[analysis.category])),node('span',match?'Matches this example':'Does not match this example',match?'match':'match mismatch'),node('p',`Recorded label: ${friendlyCondition(e.label)}`),node('p',plainExplanation(data.categories[analysis.category]),'small'),node('p',match?'The answer matches the label supplied with this recording.':'Here the computer gets the recorded label wrong. Similar vibration patterns can lead it to choose a different condition.'));$('analyze').textContent='Show the prediction again →';$('download-result').disabled=false;$('decision-panel').hidden=false;$('path-count').textContent=`${analysis.path.length} steps`;$('decision-path').replaceChildren();analysis.path.forEach(step=>{const info=featureInfo(data.features[step.f]),item=node('li');item.append(node('strong',info.title),node('code',`${format(step.v)} ${step.left?'≤':'>'} ${format(step.threshold)}`),node('small',`${step.left?'Left':'Right'} branch · ${STATS[info.stat][1]}`));$('decision-path').append(item);});table();});
  $('download-result').addEventListener('click',()=>{if(!analysis)return;const e=current();download(`rotary-analysis-row-${e.id}.json`,JSON.stringify({source:e.source,row:e.id,recordedLabel:e.label,predictedLabel:data.categories[analysis.category],matchesRecordedLabel:analysis.category===e.category,measurementSha256:e.sha256,featureTableSha256:data.featureTableSha256,model:'25 RPM time-domain decision tree',seed:data.seed,fullHoldoutAccuracy:data.accuracy,path:analysis.path.map(p=>({...p,feature:data.features[p.f]})),limitations:'Academic row-level evaluation; no independent-run validation, calibrated confidence or maintenance recommendation.'},null,2),'application/json');});
  $('download-features').addEventListener('click',()=>{const e=current(),csv=[['feature','selected_row','selected_value','no_fault_row','no_fault_value'],...data.features.map((f,i)=>[f,e.id,e.features[i],reference.id,reference.features[i]])].map(row=>row.join(',')).join('\n');download(`rotary-features-row-${e.id}.csv`,csv,'text/csv;charset=utf-8');});
  $('accuracy').textContent=`${(data.accuracy*100).toFixed(1)}%`;$('evaluation').textContent=`It correctly labelled ${Math.round(data.accuracy*data.testCount)} of ${data.testCount} test recordings, across ${Object.keys(data.categories).length} conditions. This does not mean every individual answer is ${ (data.accuracy*100).toFixed(1) }% certain.`;
  const correct=data.examples.filter(e=>e.category===predict(data.tree,e.features).category).length;$('example-score').textContent=`The 39 examples on this page include ${correct} correct answers and ${data.examples.length-correct} mistakes relative to their recorded labels. They are only part of the full test.`;
  $('table-hash').textContent=`Feature-table SHA-256: ${data.featureTableSha256}`;$('versions').textContent=`Export environment: scikit-learn ${data.versions.sklearn}, NumPy ${data.versions.numpy}. Split seed ${data.seed}. Numbers in the decision path are rounded for display; downloaded thresholds retain full precision.`;
  update();$('status').textContent='';$('demo').hidden=false;$('retry').hidden=true;
 }catch(error){$('demo').hidden=true;$('status').textContent='The measurements could not be loaded or verified. Please try again. The project and methodology links remain available below.';$('retry').hidden=false;}
}
if(typeof document!=='undefined'){$('retry').addEventListener('click',()=>location.reload());init();}
if(typeof module!=='undefined')module.exports={predict,sampleRange,validateData,featureInfo,friendlyCondition};
