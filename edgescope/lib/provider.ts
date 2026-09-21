import type {OddsEvent} from "./types";
import {FRENCH_BOOKS,REFERENCE_BOOK,scanEvent} from "./value";
const BASE="https://api.the-odds-api.com/v4";
const DEFAULT_SPORTS=["soccer_epl","soccer_france_ligue_one","soccer_uefa_champs_league","basketball_nba","icehockey_nhl"];
export async function fetchOpportunities(sport?:string){
 const apiKey=process.env.THE_ODDS_API_KEY;
 if(!apiKey)return{opportunities:[],status:"missing_api_key" as const,quota:null};
 const sports=sport?[sport]:DEFAULT_SPORTS,minEdge=Number(process.env.SCAN_EDGE_MIN??".025"),books=[...FRENCH_BOOKS,REFERENCE_BOOK].join(",");
 const all=[];let quota:null|{remaining:string|null;used:string|null;last:string|null}=null;
 for(const s of sports){const qs=new URLSearchParams({apiKey,bookmakers:books,markets:"h2h,spreads,totals",oddsFormat:"decimal",dateFormat:"iso"});const res=await fetch(`${BASE}/sports/${s}/odds?${qs}`,{cache:"no-store"});quota={remaining:res.headers.get("x-requests-remaining"),used:res.headers.get("x-requests-used"),last:res.headers.get("x-requests-last")};if(!res.ok)continue;const events=await res.json() as OddsEvent[];for(const e of events)all.push(...scanEvent(e,minEdge))}
 const unique=new Map(all.map(o=>[o.id,o]));return{opportunities:[...unique.values()].sort((a,b)=>b.edge-a.edge),status:"ok" as const,quota}
}
