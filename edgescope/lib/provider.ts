import type {OddsEvent,Opportunity} from "./types";
import {FRENCH_BOOKS,REFERENCE_BOOK,devig,scanEvent} from "./value";

const TOA_BASE="https://api.the-odds-api.com/v4";
const OP_BASE="https://api.oddspapi.io/v4";
const OP_DEFAULT_SOFT=["winamax.fr","unibet.fr","pmu","betclic.fr","netbet.fr"];
const BOOK_TITLES:Record<string,string>={"winamax.fr":"Winamax FR","unibet.fr":"Unibet FR","pmu":"PMU","betclic.fr":"Betclic FR","netbet.fr":"NetBet FR"};
const MAX_AGE=Number(process.env.MAX_ODDS_AGE_SECONDS??"180");

type OpSport={sportId:number;slug:string;sportName:string};
type OpFixture={fixtureId:string;sportId:number;tournamentId:number;startTime:string;participant1Name:string;participant2Name:string};
type OpCatalog={marketId:number;marketName:string;sportId:number;handicap:number;period:string;marketType:string;playerProp:boolean;outcomes:{outcomeId:number;outcomeName:string}[]};
type OpPrice={active:boolean;price:number;changedAt?:string;bookmakerChangedAt?:string|null;playerName?:string|null;bookmakerOutcomeId?:string};
type OpMarket={marketActive:boolean;outcomes:Record<string,{players:Record<string,OpPrice>}>};
type OpBook={bookmakerIsActive:boolean;suspended:boolean;markets:Record<string,OpMarket>};
type OpOdds={fixtureId:string;sportId:number;tournamentId:number;startTime:string;bookmakerOdds:Record<string,OpBook>};

let sportsCache:{at:number;rows:OpSport[]}|null=null;
let marketsCache:{at:number;rows:OpCatalog[]}|null=null;
const ttl=86_400_000;

async function opGet<T>(path:string,params:Record<string,string>){
 const apiKey=process.env.ODDSPAPI_API_KEY;
 if(!apiKey)throw new Error("missing_oddspapi_key");
 const qs=new URLSearchParams({...params,apiKey});
 const res=await fetch(`${OP_BASE}/${path}?${qs}`,{cache:"no-store"});
 if(!res.ok)throw new Error(`oddspapi_${res.status}`);
 return await res.json() as T;
}
async function opSports(){if(sportsCache&&Date.now()-sportsCache.at<ttl)return sportsCache.rows;const rows=await opGet<OpSport[]>("sports",{language:"en"});sportsCache={at:Date.now(),rows};return rows}
async function opMarkets(){if(marketsCache&&Date.now()-marketsCache.at<ttl)return marketsCache.rows;const rows=await opGet<OpCatalog[]>("markets",{language:"en"});marketsCache={at:Date.now(),rows};return rows}
const chunks=<T,>(a:T[],n:number)=>Array.from({length:Math.ceil(a.length/n)},(_,i)=>a.slice(i*n,(i+1)*n));
const secondsOld=(ts?:string|null)=>ts?Math.max(0,(Date.now()-Date.parse(ts))/1000):9999;
const techScore=(edge:number,age:number)=>Math.max(0,Math.min(100,Math.round(62+Math.min(22,edge*260)-Math.max(0,age-30)*.08)));
function selectionName(c:OpCatalog|undefined,outcomeId:string,player:OpPrice,f:OpFixture){
 const name=c?.outcomes.find(x=>String(x.outcomeId)===outcomeId)?.outcomeName??outcomeId;
 const normalized=c?.marketType==="1x2"?(name==="1"?f.participant1Name:name==="2"?f.participant2Name:name==="X"?"Nul":name):name;
 return player.playerName?`${player.playerName} — ${normalized}`:normalized;
}

export async function fetchOddsPapiOpportunities(sportSlug="soccer"){
 const minEdge=Number(process.env.SCAN_EDGE_MIN??".025");
 const soft=(process.env.SOFT_BOOKS??OP_DEFAULT_SOFT.join(",")).split(",").map(x=>x.trim()).filter(Boolean);
 const sports=await opSports();
 const sport=sports.find(s=>s.slug===sportSlug)||sports.find(s=>s.sportName.toLowerCase()===sportSlug.toLowerCase());
 if(!sport)return {opportunities:[] as Opportunity[],status:"sport_not_found" as const,quota:null,provider:"OddsPapi"};
 const now=new Date(),to=new Date(now.getTime()+48*3600_000);
 const fixtures=await opGet<OpFixture[]>("fixtures",{sportId:String(sport.sportId),from:now.toISOString(),to:to.toISOString(),hasOdds:"true",statusId:"0",language:"en"});
 if(!fixtures.length)return {opportunities:[] as Opportunity[],status:"ok" as const,quota:null,provider:"OddsPapi"};
 const marketCatalog=await opMarkets(),catalog=new Map(marketCatalog.map(m=>[String(m.marketId),m]));
 const fixtureMap=new Map(fixtures.map(f=>[f.fixtureId,f])),tournaments=[...new Set(fixtures.map(f=>f.tournamentId))];
 const books=["pinnacle",...soft].join(","),all:Opportunity[]=[];
 for(const part of chunks(tournaments,30)){
  const raw=await opGet<OpOdds[]|{data?:OpOdds[]}>("odds-by-tournaments",{tournamentIds:part.join(","),bookmakers:books,language:"en",verbosity:"3",oddsFormat:"decimal"});
  const rows=Array.isArray(raw)?raw:(raw.data??[]);
  for(const odds of rows){
   const f=fixtureMap.get(odds.fixtureId);if(!f)continue;
   const ref=odds.bookmakerOdds?.pinnacle;if(!ref||!ref.bookmakerIsActive||ref.suspended)continue;
   for(const [marketId,rm] of Object.entries(ref.markets??{})){
    if(!rm.marketActive)continue;
    const cat=catalog.get(marketId);
    const playerIds=new Set<string>();
    for(const o of Object.values(rm.outcomes??{}))for(const pid of Object.keys(o.players??{}))playerIds.add(pid);
    for(const pid of playerIds){
     const referenceRows=Object.entries(rm.outcomes??{}).flatMap(([outcomeId,o])=>{const p=o.players?.[pid];return p?.active&&Number(p.price)>1?[{name:outcomeId,price:Number(p.price),meta:p}]:[]});
     const fair=devig(referenceRows.map(x=>({name:x.name,price:x.price})));if(!fair)continue;
     for(const bookKey of soft){
      const b=odds.bookmakerOdds?.[bookKey],bm=b?.markets?.[marketId];if(!b||!bm||!b.bookmakerIsActive||b.suspended||!bm.marketActive)continue;
      for(const rr of referenceRows){
       const offered=bm.outcomes?.[rr.name]?.players?.[pid],fp=fair.find(x=>x.name===rr.name);
       if(!offered?.active||!fp||Number(offered.price)<=1)continue;
       const rAge=secondsOld(rr.meta.bookmakerChangedAt??rr.meta.changedAt),bAge=secondsOld(offered.bookmakerChangedAt??offered.changedAt);
       if(rAge>MAX_AGE||bAge>MAX_AGE)continue;
       const edge=fp.fairProbability*Number(offered.price)-1;if(edge<minEdge||edge>.25)continue;
       const sel=selectionName(cat,rr.name,offered,f),point=cat&&cat.handicap!==0?Number(cat.handicap):null;
       all.push({id:[f.fixtureId,marketId,rr.name,pid,bookKey].join(":"),eventId:f.fixtureId,sport:sport.sportName,event:`${f.participant1Name} — ${f.participant2Name}`,homeTeam:f.participant1Name,awayTeam:f.participant2Name,commenceTime:f.startTime,market:cat?`${cat.marketName} · ${cat.period}`:`market ${marketId}`,selection:sel,point,bookmaker:bookKey,bookmakerTitle:BOOK_TITLES[bookKey]??bookKey,offeredOdds:Number(offered.price),referenceOdds:rr.price,fairOdds:fp.fairOdds,fairProbability:fp.fairProbability,edge,confidence:techScore(edge,Math.max(rAge,bAge)),reference:"pinnacle",referenceUpdatedAt:rr.meta.bookmakerChangedAt??rr.meta.changedAt??null,bookmakerUpdatedAt:offered.bookmakerChangedAt??offered.changedAt??null});
      }
     }
    }
   }
  }
 }
 const unique=new Map(all.map(o=>[o.id,o]));
 return {opportunities:[...unique.values()].sort((a,b)=>b.edge-a.edge||b.confidence-a.confidence),status:"ok" as const,quota:null,provider:"OddsPapi"};
}

async function fetchTheOddsApiOpportunities(sport?:string){
 const apiKey=process.env.THE_ODDS_API_KEY;
 if(!apiKey)return{opportunities:[] as Opportunity[],status:"missing_api_key" as const,quota:null,provider:"The Odds API"};
 const sports=sport?[sport]:["soccer_france_ligue_one"],minEdge=Number(process.env.SCAN_EDGE_MIN??".025"),books=[...FRENCH_BOOKS,REFERENCE_BOOK].join(",");
 const all:Opportunity[]=[];let quota:null|{remaining:string|null;used:string|null;last:string|null}=null;
 for(const s of sports){const qs=new URLSearchParams({apiKey,bookmakers:books,markets:"h2h,spreads,totals",oddsFormat:"decimal",dateFormat:"iso"});const res=await fetch(`${TOA_BASE}/sports/${s}/odds?${qs}`,{cache:"no-store"});quota={remaining:res.headers.get("x-requests-remaining"),used:res.headers.get("x-requests-used"),last:res.headers.get("x-requests-last")};if(!res.ok)continue;const events=await res.json() as OddsEvent[];for(const e of events)all.push(...scanEvent(e,minEdge))}
 const unique=new Map(all.map(o=>[o.id,o]));return{opportunities:[...unique.values()].sort((a,b)=>b.edge-a.edge),status:"ok" as const,quota,provider:"The Odds API"};
}

export async function fetchOpportunities(sport?:string){
 if(process.env.ODDSPAPI_API_KEY)return fetchOddsPapiOpportunities(sport??"soccer");
 return fetchTheOddsApiOpportunities(sport);
}
