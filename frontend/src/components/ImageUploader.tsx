import React, { useRef, useState, useCallback } from 'react';
import { Upload } from 'lucide-react';

interface Props {
  onImageSelect: (file: File) => void;
}

const ImageUploader: React.FC<Props> = ({ onImageSelect }) => {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelect(e.dataTransfer.files[0]);
    }
  }, [onImageSelect]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onImageSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className={`border border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 ${
        dragOver 
          ? 'border-rose-500 bg-rose-50 dark:bg-rose-950/20' 
          : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-50/50 dark:bg-slate-950/50'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} className="hidden" />
      <div className="flex flex-col items-center justify-center space-y-3">
        <Upload size={24} className={dragOver ? 'text-rose-500' : 'text-slate-400'} />
        <p className="text-[11px] text-slate-500 uppercase tracking-wider">
          <span className="sm:hidden">Tap to capture</span>
          <span className="hidden sm:inline">Drop image or browse</span>
        </p>
      </div>
    </div>
  );
};

export default ImageUploader;
