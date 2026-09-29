import {NextRequest,NextResponse} from "next/server";import {fetchOpportunities} from "@/lib/provider";import {persistOpportunities} from "@/lib/persist";
export const runtime="nodejs";export const dynamic="force-dynamic";
export async function POST(req:NextRequest){
 const secret=process.env.CRON_SECRET;if(!secret)return NextResponse.json({status:"missing_cron_secret"},{status:503});
 const auth=req.headers.get("authorization");if(auth!==`Bearer ${secret}`)return NextResponse.json({status:"unauthorized"},{status:401});
 const scan=await fetchOpportunities();if(scan.status!=="ok")return NextResponse.json(scan,{status:503});
 const saved=await persistOpportunities(scan.opportunities);return NextResponse.json({status:saved.status,found:scan.opportunities.length,inserted:saved.inserted,alerts:saved.alerts,quota:scan.quota});
}
