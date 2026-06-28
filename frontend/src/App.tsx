import { Routes, Route } from "react-router-dom";  // <-- use react-router-dom
import Layout from "@/components/Layout";
import Dashboard from "@/pages/Dashboard";
import Predictor from "@/pages/Predictor";
import Training from "@/pages/Training";
import Results from "@/pages/Results";
import Explain from "@/pages/Explain";
import Poster from "@/pages/Poster";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/predictor" element={<Predictor />} />
        <Route path="/training" element={<Training />} />
        <Route path="/results" element={<Results />} />
        <Route path="/explain" element={<Explain />} />
        <Route path="/poster" element={<Poster />} />
      </Routes>
    </Layout>
  );
}
