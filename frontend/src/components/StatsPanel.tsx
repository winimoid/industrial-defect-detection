import { Target, Zap, ShieldCheck, AlertCircle } from 'lucide-react';

export default function StatsPanel({ confidence, defectType, inferenceTime, defective }: any) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <StatCard 
        icon={<Target size={14} />}
        label="Confidence" 
        value={`${(confidence * 100).toFixed(1)}%`} 
      />
      <StatCard 
        icon={defective ? <AlertCircle size={14}/> : <ShieldCheck size={14}/>}
        label="Analysis" 
        value={defective ? defectType : 'Pass'} 
        highlight={defective}
      />
      <StatCard 
        icon={<Zap size={14}/>}
        label="Inference" 
        value={`${inferenceTime.toFixed(1)}ms`} 
      />
    </div>
  );
}

function StatCard({ icon, label, value, highlight }: any) {
  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 rounded-2xl shadow-sm">
      <div className="flex items-center gap-2 mb-2 text-slate-400">
        {icon}
        <p className="text-[10px] uppercase tracking-[0.2em]">{label}</p>
      </div>
      <p className={`text-2xl font-light tracking-tight ${highlight ? 'text-rose-500' : 'dark:text-white'}`}>
        {value}
      </p>
    </div>
  );
}
