import type {Metadata} from "next";import "./globals.css";
export const metadata:Metadata={title:"EdgeScope — Value Scanner",description:"Scanner quantitatif de value betting basé sur des prix de marché."};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="fr"><body>{children}</body></html>}
