import type {Opportunity} from "./types";import {adminDb} from "./supabase-admin";import {sendTelegram} from "./telegram";
export async function persistOpportunities(rows:Opportunity[]){
 const db=adminDb();if(!db)return {status:"missing_supabase_secret" as const,inserted:0,alerts:0};
 let inserted=0,alerts=0;
 for(const o of rows){
  const {data:event,error:eventError}=await db.from("events").upsert({provider_event_id:o.eventId,sport_key:o.sport.toLowerCase().replace(/\s+/g,"_"),sport_title:o.sport,home_team:o.homeTeam,away_team:o.awayTeam,commence_time:o.commenceTime,updated_at:new Date().toISOString()},{onConflict:"provider_event_id"}).select("id").single();
  if(eventError||!event)continue;
  const payload={fingerprint:o.id,event_id:event.id,sport_key:o.sport,event_name:o.event,market_key:o.market,selection:o.selection,point:o.point,bookmaker_key:o.bookmaker,offered_odds:o.offeredOdds,reference_key:o.reference,reference_odds:o.referenceOdds,fair_odds:o.fairOdds,fair_probability:o.fairProbability,edge:o.edge,confidence:o.confidence,status:"active",engine_version:"1.0.0"};
  const {data:existing}=await db.from("opportunities").select("id").eq("fingerprint",o.id).maybeSingle();
  const {error}=await db.from("opportunities").upsert(payload,{onConflict:"fingerprint"});
  if(error)continue;if(!existing)inserted++;
  if(!existing&&o.confidence>=70){
    const result=await sendTelegram(o);
    if(result.sent){const {data:opp}=await db.from("opportunities").select("id").eq("fingerprint",o.id).single();if(opp){await db.from("alerts").upsert({opportunity_id:opp.id,channel:"telegram"},{onConflict:"opportunity_id,channel"});alerts++}}
  }
 }
 return {status:"ok" as const,inserted,alerts};
}
