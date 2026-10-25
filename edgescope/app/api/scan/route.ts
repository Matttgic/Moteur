import {NextRequest,NextResponse} from "next/server";
import {fetchOpportunities} from "@/lib/provider";
export const runtime="nodejs";export const dynamic="force-dynamic";
export async function GET(req:NextRequest){try{const sport=req.nextUrl.searchParams.get("sport")??undefined;const data=await fetchOpportunities(sport);return NextResponse.json({...data,generatedAt:new Date().toISOString()})}catch(error){return NextResponse.json({status:"error",error:error instanceof Error?error.message:"unknown_error"},{status:500})}}
