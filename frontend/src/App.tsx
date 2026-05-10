import { useState, useEffect } from 'react';
import { 
  Sun, Moon, Cpu, Globe, Database, GitBranch, 
  Activity, Layout, Play, RefreshCw 
} from 'lucide-react';
import ImageUploader from './components/ImageUploader';
import ResultOverlay from './components/ResultOverlay';
import StatsPanel from './components/StatsPanel';

// --- CONSTANTS FOR API CALLs ---
const API_URL = '';  // we use relative path, gets proxied by nginx
// const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const DEMO_IMAGES = [
  { name: 'Defect 1', src: '/demo/defect1.png' },
  { name: 'Defect 2', src: '/demo/defect2.png' },
  { name: 'Defect 3', src: '/demo/defect3.png' },
  { name: 'Good 1', src: '/demo/good1.png' },
  { name: 'Good 2', src: '/demo/good2.png' },
];

interface Result {
  defective: boolean;
  confidence: number;
  defect_type: string | null;
  bbox: number[];
  inference_time_ms: number;
}

export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [selectedImageUrl, setSelectedImageUrl] = useState<string | null>(null);

  // Dark Mode Toggle Logic
  useEffect(() => {
    if (darkMode) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [darkMode]);

  // --- then the API call ---
  const handleImage = async (file: File) => {
    const url = URL.createObjectURL(file);
    setSelectedImageUrl(url);
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      const data: Result = await res.json();
      setResult(data);
    } catch (err) {
      console.error("API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  // --- with a little demo ---
  const handleDemo = async (src: string) => {
    const response = await fetch(src);
    const blob = await response.blob();
    const file = new File([blob], src.split('/').pop() || 'demo.png', { type: blob.type });
    handleImage(file);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 industrial-grid text-slate-900 dark:text-slate-100 transition-colors duration-300 font-sans">
      
      {/* HEADER */}
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-rose-500 rounded-lg flex items-center justify-center text-white font-bold italic shadow-lg shadow-rose-500/20">V</div>
            <h1 className="text-sm md:text-lg font-semibold tracking-tight uppercase flex gap-2">
              Vision <span className="font-light opacity-50 hidden sm:inline">Industrial Defect Detection</span>
            </h1>
          </div>
          
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all"
            >
              {darkMode ? <Sun size={18} className="text-yellow-500" /> : <Moon size={18} className="text-slate-600" />}
            </button>
            <a href="https://github.com/winimoid/industrial-defect-detection" title="GitHub Repository" target="_blank" rel="noopener noreferrer" className="p-2 opacity-50 hover:opacity-100 transition-opacity">
              <GitBranch size={20} />
            </a>
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="max-w-[1400px] mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: Controls & Demos */}
        <div className="lg:col-span-4 space-y-6 order-2 lg:order-1">
          <section className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <Layout size={14} className="text-rose-500" />
              <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Input Source</h2>
            </div>
            
            <ImageUploader onImageSelect={handleImage} />

            <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-800">
              <div className="flex items-center justify-center gap-2 mb-4">
                <Play size={12} className="text-slate-400" />
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Sample Dataset</p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {DEMO_IMAGES.map((img) => (
                  <button
                    key={img.src}
                    onClick={() => handleDemo(img.src)}
                    className="px-3 py-1.5 text-[11px] bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg hover:border-rose-500 transition-all flex items-center gap-2"
                  >
                    <div className="w-1 h-1 rounded-full bg-slate-400"></div>
                    {img.name}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <Activity size={14} className="text-rose-500" />
              <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">System Pipeline</h2>
            </div>
            <div className="space-y-3">
              <StatusItem icon={<Cpu size={14}/>} label="Model" value="YOLOv8n" />
              <StatusItem icon={<Database size={14}/>} label="Optimization" value="ONNX INT8" />
              <StatusItem icon={<Globe size={14}/>} label="Deployment" value="Edge Container" status="Online" />
            </div>
          </section>
        </div>

        {/* RIGHT COLUMN: Results Display */}
        <div className="lg:col-span-8 order-1 lg:order-2">
          {!selectedImageUrl ? (
            <div className="h-[400px] md:h-[600px] flex flex-col items-center justify-center border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-3xl opacity-30">
              <RefreshCw size={48} className="mb-4 animate-spin-slow opacity-20" />
              <p className="font-light tracking-widest uppercase text-xs text-center">Awaiting data input...</p>
            </div>
          ) : (
            <div className="space-y-6">
              {result && (
                <StatsPanel 
                  confidence={result.confidence} 
                  defectType={result.defect_type} 
                  inferenceTime={result.inference_time_ms} 
                  defective={result.defective} 
                />
              )}
              
              <div className="relative bg-white dark:bg-slate-900 rounded-3xl overflow-hidden border border-slate-200 dark:border-slate-800 shadow-xl min-h-[300px]">
                {loading && (
                  <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm z-20 flex items-center justify-center">
                    <div className="text-center">
                      <RefreshCw className="w-10 h-10 text-rose-500 animate-spin mx-auto mb-4" />
                      <p className="text-white font-light tracking-widest uppercase text-[10px]">Processing Vision Matrix...</p>
                    </div>
                  </div>
                )}
                <ResultOverlay 
                  imageUrl={selectedImageUrl} 
                  bbox={result?.bbox || []} 
                  defective={result?.defective || false} 
                  type={result?.defect_type || null}
                />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function StatusItem({ icon, label, value, status }: any) {
  return (
    <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-800 group hover:border-slate-300 dark:hover:border-slate-600 transition-colors">
      <div className="flex items-center gap-3">
        <span className="text-slate-400 group-hover:text-rose-500 transition-colors">{icon}</span>
        <span className="text-[11px] font-medium text-slate-500 uppercase">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        {status && <div className="w-1 h-1 bg-emerald-500 rounded-full animate-pulse"></div>}
        <span className="text-[11px] font-mono text-slate-700 dark:text-slate-300">{value}</span>
      </div>
    </div>
  );
}
