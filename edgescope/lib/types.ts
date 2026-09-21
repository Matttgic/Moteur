export type Outcome={name:string;price:number;point?:number};
export type Market={key:string;last_update?:string;outcomes:Outcome[]};
export type Bookmaker={key:string;title:string;last_update?:string;markets:Market[]};
export type OddsEvent={id:string;sport_key:string;sport_title:string;commence_time:string;home_team:string;away_team:string;bookmakers:Bookmaker[]};
export type Opportunity={id:string;eventId:string;sport:string;event:string;commenceTime:string;market:string;selection:string;point:number|null;bookmaker:string;bookmakerTitle:string;offeredOdds:number;referenceOdds:number;fairOdds:number;fairProbability:number;edge:number;confidence:number;reference:string;referenceUpdatedAt:string|null;bookmakerUpdatedAt:string|null};
