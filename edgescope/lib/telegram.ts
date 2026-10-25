import type {Opportunity} from "./types";
export async function sendTelegram(o:Opportunity){
 const token=process.env.TELEGRAM_BOT_TOKEN,chatId=process.env.TELEGRAM_CHAT_ID;
 if(!token||!chatId)return {sent:false,reason:"not_configured"};
 const text=[
  "🚨 EDGE DETECTED",`🏟 ${o.event}`,`🎯 ${o.selection}${o.point==null?"":` ${o.point}`}`,
  `🏦 ${o.bookmakerTitle} @ ${o.offeredOdds.toFixed(2)}`,`📊 Fair: ${o.fairOdds.toFixed(2)}`,
  `⚡ Edge: ${(o.edge*100).toFixed(2)}%`,`🧪 Confiance technique: ${o.confidence}/100`,"Référence: Pinnacle"
 ].join("\n");
 const res=await fetch(`https://api.telegram.org/bot${token}/sendMessage`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({chat_id:chatId,text})});
 return {sent:res.ok,status:res.status};
}
