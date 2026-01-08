import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { TrackData, AVAILABLE_MODELS } from '../types';

interface TimeSeriesChartProps {
  trackData: TrackData;
  dataKey: 'windSpeed' | 'pressure';
  title: string;
  unit: string;
  color: string;
}

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({ trackData, dataKey, title, unit, color }) => {
  
  // We need a unified timeline.
  const timeMap = new Map<number, any>();
  const models = Object.keys(trackData).filter(k => k !== 'past');
  
  const addToMap = (data: any[], keyName: string) => {
      data.forEach(p => {
          if (p[dataKey] === null || p[dataKey] === undefined) return;
          const existing = timeMap.get(p.time) || { time: p.time };
          existing[keyName] = p[dataKey];
          timeMap.set(p.time, existing);
      });
  };

  addToMap(trackData.past, 'past');
  models.forEach(m => addToMap(trackData[m], m));

  // Connect past to models if possible
  const lastPastPoint = trackData.past[trackData.past.length - 1];
  if (lastPastPoint && lastPastPoint[dataKey] !== null) {
      const connect = timeMap.get(lastPastPoint.time);
      if (connect) {
          models.forEach(m => {
              // Only connect if the model actually has data
              // We check if the FIRST point of the model has data? 
              // Simplification: just assign past value to model start key for continuity if model exists
              if (trackData[m].length > 0 && trackData[m][0][dataKey] !== null) {
                connect[m] = lastPastPoint[dataKey];
              }
          });
      }
  }

  const chartData = Array.from(timeMap.values()).sort((a, b) => a.time - b.time);

  // Determine line color
  const getLineColor = (key: string) => {
      if (key === 'past') return '#94a3b8';
      if (key === 'groundTruth') return '#22c55e';
      if (key === 'sfdiff') return '#f43f5e';
      const m = AVAILABLE_MODELS.find(mod => mod.id === key);
      return m ? m.color : '#888888';
  };

  return (
    <div className="bg-surface/50 backdrop-blur-md border border-white/10 rounded-xl p-4 shadow-xl h-[300px] flex flex-col">
      <h3 className="text-md font-bold text-gray-200 mb-2">{title}</h3>
      <div className="flex-grow w-full text-xs">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" domain={['auto', 'auto']}/>
            <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff' }}
                itemStyle={{ color: '#fff' }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(value: number) => [`${value.toFixed(1)} ${unit}`, '']}
            />
            <Legend />
            <Line type="monotone" dataKey="past" stroke={getLineColor('past')} strokeWidth={2} dot={false} name="Past" connectNulls />
            <Line type="monotone" dataKey="groundTruth" stroke={getLineColor('groundTruth')} strokeWidth={2} dot={false} name="Truth" connectNulls />
            
            {models.map(m => {
                if (m === 'groundTruth' || m === 'past') return null;
                // Check if this model has any data for this key
                const hasData = chartData.some(d => d[m] !== undefined);
                if (!hasData) return null;

                return (
                    <Line 
                        key={m}
                        type="monotone" 
                        dataKey={m} 
                        stroke={getLineColor(m)} 
                        strokeWidth={m === 'sfdiff' ? 3 : 2} 
                        strokeDasharray={m === 'sfdiff' ? undefined : "5 5"}
                        dot={m === 'sfdiff' ? {r: 3} : false} 
                        name={m.toUpperCase()} 
                        connectNulls 
                    />
                );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TimeSeriesChart;