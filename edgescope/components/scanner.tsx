"use client";
import {useCallback,useEffect,useState} from "react";import type {Opportunity} from "@/lib/types";
type Payload={status:string;provider?:string;generatedAt?:string;opportunities?:Opportunity[];quota?:{remaining:string|null;used:string|null;last:string|null}|null};
const pct=(n:number)=>`${(n*100).toFixed(2)}%`,odds=(n:number)=>n.toFixed(2);
export default function Scanner(){
 const[data,setData]=useState<Payload>({status:"loading"}),[busy,setBusy]=useState(false),[sport,setSport]=useState("soccer");
 const load=useCallback(async(s=sport)=>{setBusy(true);try{const r=await fetch(`/api/scan?sport=${encodeURIComponent(s)}`,{cache:"no-store"});setData(await r.json())}finally{setBusy(false)}},[sport]);
 useEffect(()=>{load(sport)},[sport,load]);
 const rows=data.opportunities??[];
 return <main className="wrap"><div className="top"><div><div className="brand">EdgeScope</div><div className="sub">Pinnacle fair price → bookmakers FR · {data.provider??"source à connecter"}</div></div><button onClick={()=>load()} disabled={busy}>{busy?"Scan…":"Actualiser"}</button></div>
 <nav className="nav"><a href="/">Scanner</a><a href="/history">Historique</a><a href="/performance">Performance</a></nav>
 <div className="toolbar"><label className="sub">Sport <select value={sport} onChange={e=>setSport(e.target.value)}><option value="soccer">Football</option><option value="basketball">Basketball</option><option value="tennis">Tennis</option></select></label><span className="sub">Actualisation manuelle pendant le test pour économiser le quota API.</span></div>
 <section className="grid"><div className="metric"><span className="sub">Opportunités</span><b>{rows.length}</b></div><div className="metric"><span className="sub">Meilleur edge</span><b>{rows[0]?pct(rows[0].edge):"—"}</b></div><div className="metric"><span className="sub">Seuil</span><b>2.5%</b></div><div className="metric"><span className="sub">Quota</span><b>{data.quota?.remaining??"—"}</b></div></section>
 {data.status==="missing_api_key"?<div className="card empty">Ajoute une clé OddsPapi (prioritaire) ou The Odds API pour activer les vraies cotes.</div>:null}
 {data.status==="sport_not_found"?<div className="card empty">Sport non trouvé par le fournisseur.</div>:null}
 {data.status==="ok"&&rows.length===0?<div className="card empty">Aucune value ≥ 2,5% suffisamment fraîche actuellement.</div>:null}
 <section className="list">{rows.map(o=><article className="card row" key={o.id}><div><b>{o.event}</b><div className="sub">{o.sport} · {o.market} {o.point??""}</div></div><div><span className="pill">{o.bookmakerTitle}</span><div><b>{o.selection}</b> @ {odds(o.offeredOdds)}</div></div><div><div className="sub">Fair</div><b>{odds(o.fairOdds)}</b></div><div><div className="sub">Edge</div><b className="edge">{pct(o.edge)}</b></div><div><div className="sub">Confiance</div><b>{o.confidence}/100</b></div></article>)}</section>
 <p className="sub">18+ · Aucun gain garanti. “Confiance” mesure la qualité technique du signal, pas sa probabilité de gagner.</p></main>
}