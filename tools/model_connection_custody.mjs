import {randomUUID} from 'node:crypto';

async function targetId(page){
 const session=await page.createCDPSession();
 try{return (await session.send('Target.getTargetInfo')).targetInfo.targetId;}
 finally{await session.detach();}
}

const geometry=page=>page.evaluate(()=>({width:window.innerWidth,height:window.innerHeight,
 deviceScaleFactor:window.devicePixelRatio}));
const validGeometry=v=>Number.isSafeInteger(v?.width)&&v.width>0&&Number.isSafeInteger(v?.height)&&v.height>0&&
 Number.isFinite(v.deviceScaleFactor)&&v.deviceScaleFactor>0;
const sameViewport=(a,b)=>['width','height','deviceScaleFactor','isMobile','hasTouch','isLandscape'].every(key=>
 (a?.[key]??({deviceScaleFactor:1,isMobile:false,hasTouch:false,isLandscape:false}[key]))===
 (b?.[key]??({deviceScaleFactor:1,isMobile:false,hasTouch:false,isLandscape:false}[key])));

// Initial navigation/status/mount/render proof stays on the ordinary connection.
// Only unused model-phase CDP network body duplication is disabled. No reload,
// browser close, new target, host replacement or payload truncation is allowed.
export async function reconnectModelWithoutNetworkCopies({puppeteer,browser,page,endpoint}){
 if(browser.wsEndpoint()!==endpoint||page.browser()!==browser)throw Error('exact owned browser endpoint/page custody required');
 const url=page.url(),before=await targetId(page),nonce=randomUUID();
 const viewportBefore=structuredClone(page.viewport()),geometryBefore=await geometry(page);
 if(!Number.isSafeInteger(viewportBefore?.width)||viewportBefore.width<=0||
  !Number.isSafeInteger(viewportBefore?.height)||viewportBefore.height<=0||!validGeometry(geometryBefore))
  throw Error('explicit witnessed viewport/geometry custody required');
 const receipt={route:'exact-owned-model-cdp-network-disabled',requestedNetworkEnabled:false,
  effectiveNetworkEnabled:null,targetBefore:before,urlBefore:url,viewportBefore,geometryBefore,
  initialChecks:'retained before reconnect',status:'reconnecting'};
 let connection,selected,marked=false;
 const clear=async p=>p.evaluate(nonce=>{if(window.__miniModelConnectionCustody?.nonce===nonce)delete window.__miniModelConnectionCustody;},nonce);
 try{
  await page.evaluate(nonce=>{
   if(window.__miniModelConnectionCustody)throw Error('owned model connection marker already exists');
   window.__miniModelConnectionCustody={nonce,foreground:window.__miniResidentForeground,
    volume:window.__kaminosVolumePrototype,setup:window.__kaminosCompositionSetup};
  },nonce);marked=true;
  browser.disconnect();
  connection=await puppeteer.connect({browserWSEndpoint:endpoint,protocolTimeout:0,networkEnabled:false,
   defaultViewport:structuredClone(viewportBefore)});
  if(connection.wsEndpoint()!==endpoint)throw Error('reconnected owned browser endpoint changed');
  const pages=await connection.pages();
  if(pages.length!==1)throw Error('exact owned single model page custody required');
  selected=pages[0];receipt.targetAfter=await targetId(selected);receipt.urlAfter=selected.url();
  if(receipt.targetAfter!==before||receipt.urlAfter!==url||selected.browser()!==connection)throw Error('owned model target/URL custody changed');
  receipt.viewportAfter=structuredClone(selected.viewport());receipt.geometryAfter=await geometry(selected);
  if(!sameViewport(viewportBefore,receipt.viewportAfter)||!validGeometry(receipt.geometryAfter)||
   ['width','height','deviceScaleFactor'].some(key=>geometryBefore[key]!==receipt.geometryAfter[key]))
   throw Error('owned model witnessed viewport/geometry changed');
  receipt.hostReferencesPreserved=await selected.evaluate(nonce=>{
   const state=window.__miniModelConnectionCustody;
   return state?.nonce===nonce&&state.foreground===window.__miniResidentForeground&&
    state.volume===window.__kaminosVolumePrototype&&state.setup===window.__kaminosCompositionSetup;
  },nonce);
  if(!receipt.hostReferencesPreserved)throw Error('owned model live global/host custody changed');
  await clear(selected);marked=false;
  receipt.status='connected';receipt.effectiveNetworkEnabled=false;
  receipt.evidenceLimit='Puppeteer HTTPRequest/HTTPResponse unavailable during model phase; actual fetch/status/raw receiver unchanged';
  return{browser:connection,page:selected,receipt};
 }catch(error){
  receipt.status='failed';receipt.error=error.message;
  if(marked){try{await clear(selected??page);}catch(cleanup){receipt.markerCleanupError=cleanup.message;}}
  if(connection)connection.disconnect();
  error.modelConnectionReceipt=receipt;throw error;
 }
}
