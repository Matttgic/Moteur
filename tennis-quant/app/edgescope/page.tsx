import EdgeScopeClient from "./EdgeScopeClient";

export const metadata = {
  title: "EdgeScope — Multi-sports Value Scanner",
  description: "Scanner multi-sports de value betting basé sur les prix Pinnacle et les bookmakers FR."
};

export default function EdgeScopePage() {
  return <EdgeScopeClient />;
}
