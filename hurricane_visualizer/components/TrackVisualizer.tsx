import React, { useEffect, useRef } from 'react';
import { TrackData, AVAILABLE_MODELS } from '../types';

interface TrackVisualizerProps {
  trackData: TrackData;
}

const TrackVisualizer: React.FC<TrackVisualizerProps> = ({ trackData }) => {
  const mapRef = useRef<any>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const layersRef = useRef<any[]>([]);

  // Function to get color based on model
  const getColor = (type: string) => {
    if (type === 'past') return '#94a3b8';
    if (type === 'groundTruth') return '#22c55e';
    if (type === 'sfdiff') return '#f43f5e';
    
    const model = AVAILABLE_MODELS.find(m => m.id === type);
    return model ? model.color : '#ffffff';
  };

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize Map if not exists
    if (mapRef.current === null) {
      // @ts-ignore
      const L = window.L;
      if (!L) return;

      const map = L.map(mapContainerRef.current, {
        zoomControl: false,
        attributionControl: false
      }).setView([25, -70], 4);
      
      // Satellite Tiles (Esri World Imagery) for "Image" look
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 17
      }).addTo(map);

      // Add labels/borders on top (Stamen or Carto labels)
      L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd',
        maxZoom: 19,
        opacity: 0.6
      }).addTo(map);

      L.control.zoom({ position: 'topright' }).addTo(map);

      mapRef.current = map;
    }

    const map = mapRef.current;
    // @ts-ignore
    const L = window.L;

    // Invalidate size to ensure it renders correctly after container mount
    setTimeout(() => {
        map.invalidateSize();
    }, 100);

    // Clear existing layers
    layersRef.current.forEach(layer => map.removeLayer(layer));
    layersRef.current = [];

    // Helper to draw track
    const drawTrack = (data: any[], type: string) => {
      if (!data || data.length === 0) return;
      
      const latlngs = data.map(d => [d.lat, d.lon]);
      const color = getColor(type);
      const isMain = type === 'sfdiff' || type === 'groundTruth';

      // Polyline
      const polyline = L.polyline(latlngs, {
        color: color,
        weight: isMain ? 3 : 2,
        opacity: isMain ? 1 : 0.7,
        dashArray: (type === 'past' || isMain) ? null : '5, 5'
      }).addTo(map);
      
      const popupContent = `<strong style="color:${color}">${type.toUpperCase()}</strong>`;
      polyline.bindPopup(popupContent);
      layersRef.current.push(polyline);

      // Dots for timesteps
      data.forEach((d: any, idx: number) => {
        // Only draw dots for every other point for cleaner look on dense tracks, unless short
        if (data.length > 15 && idx % 2 !== 0) return;

        const marker = L.circleMarker([d.lat, d.lon], {
          radius: isMain ? 4 : 3,
          fillColor: color,
          color: "#000",
          weight: 1,
          opacity: 1,
          fillOpacity: 1
        }).addTo(map);
        
        marker.bindPopup(`
          <div style="color: #1e293b; font-family: sans-serif; font-size: 12px;">
            <strong style="color:${color}">${type.toUpperCase()}</strong><br/>
            T: +${d.time}h<br/>
            Pos: ${d.lat.toFixed(2)}, ${d.lon.toFixed(2)}<br/>
            ${d.windSpeed ? `Wind: ${d.windSpeed} kt<br/>` : ''}
            ${d.pressure ? `Press: ${d.pressure} mb` : ''}
          </div>
        `);
        layersRef.current.push(marker);
      });
    };

    // Draw all available tracks in data
    Object.keys(trackData).forEach(key => {
        drawTrack(trackData[key], key);
    });

    // Fit bounds
    const allPoints = Object.values(trackData).flat();
    if (allPoints.length > 0) {
        const bounds = L.latLngBounds(allPoints.map(p => [p.lat, p.lon]));
        map.fitBounds(bounds, { padding: [50, 50] });
    }

  }, [trackData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  return (
    <div className="bg-surface/50 backdrop-blur-md border border-white/10 rounded-xl p-4 shadow-xl h-[600px] flex flex-col">
      <div className="mb-4 flex justify-between items-center">
        <h3 className="text-xl font-bold text-gray-100 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.21 14.37a8.21 8.21 0 0 1 .49-2.73C5.64 5.25 10.09 1 12 1c1.92 0 6.36 4.25 8.3 10.63.15.48.27.97.35 1.47A9.06 9.06 0 0 1 12 21a9.06 9.06 0 0 1-8.79-6.63Z" />
            </svg>
            World Map Forecast
        </h3>
        <div className="flex gap-2 text-xs flex-wrap justify-end">
             <span className="flex items-center gap-1 bg-black/30 px-2 py-1 rounded"><span className="w-2 h-2 rounded-full bg-slate-400"></span> Obs</span>
             <span className="flex items-center gap-1 bg-black/30 px-2 py-1 rounded"><span className="w-2 h-2 rounded-full bg-green-500"></span> Truth</span>
             <span className="flex items-center gap-1 bg-black/30 px-2 py-1 rounded"><span className="w-2 h-2 rounded-full bg-red-500"></span> SFDiff</span>
        </div>
      </div>
      
      <div className="flex-grow w-full rounded-lg overflow-hidden border border-white/5 relative bg-black">
        <div ref={mapContainerRef} className="w-full h-full z-10" />
      </div>
    </div>
  );
};

export default TrackVisualizer;