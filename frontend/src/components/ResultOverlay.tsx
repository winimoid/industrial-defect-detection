import { useEffect, useRef } from 'react';

export default function ResultOverlay({ imageUrl, bbox, defective, type }: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  console.log("Rendering ResultOverlay with:", { imageUrl, bbox, defective, type });
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      if (defective && bbox.length === 4) {
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 4;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        
        ctx.fillStyle = '#f43f5e';
        ctx.font = 'bold 24px sans-serif';
        ctx.fillText(`${type?.toUpperCase() || 'DEFECT'}`, bbox[0], bbox[1] - 10);
      }
    };
    img.src = imageUrl;
  }, [type, imageUrl, bbox, defective]);

  return (
    <div className="w-full flex justify-center p-2">
      <canvas
        ref={canvasRef}
        className="w-full h-auto max-h-[70vh] object-contain rounded-xl"
      />
    </div>
  );
}
