/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { 
  Upload, Play, Square, Activity, 
  Video, Image as ImageIcon, Box, Terminal,
  Cpu, AlertCircle, CheckCircle2, LayoutDashboard,
  Settings, History, BarChart2, X, Lock, User, LogOut, PieChart as PieChartIcon,
  Camera, Usb, Users, Trash2, Menu, Download, FileSpreadsheet, FileJson2,
  Images, ChevronDown, Loader2
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';
import { authApi, recordsApi, usersApi, inferenceApi, type UserProfile } from './api';
import { ToastContainer, toast } from './components/Toast';

type MediaType = 'image' | 'video' | null;

interface Detection {
  classId: number;
  className: string;
  score: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
}

interface LogEntry {
  id: string;
  time: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'warning';
}

interface HistoryRecord {
  id: string;
  time: string;
  image: string;
  stats: { total: number; fps: number; avgConf: number };
  detections: Detection[];
}

// Custom Classes
const CUSTOM_CLASSES = [
  "person", "rider", "car", "bus", "truck", "bike", "motorcycle", "traffic light", "traffic sign", "train"
];

const COLORS = [
  '#00daf3', '#ffb692', '#ffdad6', '#9cf0ff', '#d7e2ff', '#f06600', '#009fb2'
];

export default function App() {
  // State
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState<string>('未加载模型');
  const [modelProvider, setModelProvider] = useState<string>('');
  const [mediaFile, setMediaFile] = useState<File | null>(null);
  const [mediaType, setMediaType] = useState<MediaType>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  
  const [isInferencing, setIsInferencing] = useState(false);
  const [confThreshold, setConfThreshold] = useState(0.45);
  const [iouThreshold, setIouThreshold] = useState(0.50);
  const [targetFps, setTargetFps] = useState(60);
  const targetFpsRef = useRef<number>(60);
  targetFpsRef.current = targetFps;
  
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [stats, setStats] = useState({ total: 0, fps: 0, avgConf: 0 });
  const [activeTab, setActiveTab] = useState<'dashboard' | 'history' | 'analytics' | 'settings'>('dashboard');
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null);
  
  // Auth State
  const [isAuthenticated, setIsAuthenticated] = useState(authApi.isLoggedIn());
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [userId, setUserId] = useState('');
  
  // Media Source State
  const [sourceMode, setSourceMode] = useState<'file' | 'camera' | 'usb'>('file');
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [usersList, setUsersList] = useState<UserProfile[]>([]);

  // Enhanced UI State
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);
  const latestDetectionsRef = useRef<Detection[]>([]);
  const videoSessionIdRef = useRef<string>('');
  const videoSessionStartRef = useRef<number>(0);
  const mediaTypeRef = useRef<MediaType>(null);
  const sourceModeRef = useRef<'file' | 'camera' | 'usb'>('file');
  const isInferencingRef = useRef<boolean>(false);
  const modelLoadedRef = useRef<boolean>(false);
  const fpsHistoryRef = useRef<number[]>([]);
  const lastMeasuredFpsRef = useRef<number>(0);
  mediaTypeRef.current = mediaType;
  sourceModeRef.current = sourceMode;
  isInferencingRef.current = isInferencing;
  modelLoadedRef.current = modelLoaded;

// Stats history for sparkline (last N detection counts)
  const [detectionHistory, setDetectionHistory] = useState<number[]>([]);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const now = new Date();
    const time = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    setLogs(prev => [{ id: Math.random().toString(36).substr(2, 9), time, message, type }, ...prev].slice(0, 50));
  }, []);

  const captureSnapshot = useCallback((source: HTMLVideoElement | HTMLImageElement, overlay: HTMLCanvasElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = overlay.width;
    canvas.height = overlay.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    ctx.drawImage(overlay, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);

  // --- Camera & Devices ---
  const getDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter(device => device.kind === 'videoinput');
      setVideoDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch (err) {
      console.error("Error enumerating devices:", err);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    getDevices();
    navigator.mediaDevices.addEventListener('devicechange', getDevices);
    return () => navigator.mediaDevices.removeEventListener('devicechange', getDevices);
  }, [getDevices]);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const streamRequestRef = useRef<number>(0);

  const startStream = async (deviceId: string, useDefaultCamera: boolean = false) => {
    stopStream();
    const requestId = ++streamRequestRef.current;
    
    // Add a small delay to allow the OS to fully release the previous camera
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      const constraints: MediaStreamConstraints = {
        video: useDefaultCamera 
          ? { facingMode: 'user' } 
          : (deviceId ? { deviceId: { ideal: deviceId } } : true)
      };
      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // If another stream was requested while we were waiting, stop this one
      if (requestId !== streamRequestRef.current) {
        newStream.getTracks().forEach(track => track.stop());
        return;
      }
      
      streamRef.current = newStream;
      setStream(newStream);
      setMediaType('video');
      setMediaUrl('stream'); 
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
      }
      addLog(`已连接${useDefaultCamera ? '摄像头' : '视频设备'}`, 'info');
      getDevices();
    } catch (err: any) {
      if (requestId === streamRequestRef.current) {
        addLog(`无法访问设备: ${err.message}`, 'error');
      }
    }
  };

  const switchToMode = (mode: 'file' | 'camera' | 'usb') => {
    setSourceMode(mode);
    setIsInferencing(false);
    if (mode === 'file') {
      stopStream();
      setMediaType(mediaFile ? (mediaFile.type.startsWith('video/') ? 'video' : 'image') : null);
      setMediaUrl(mediaFile ? URL.createObjectURL(mediaFile) : null);
    } else if (mode === 'camera') {
      startStream('', true);
    } else if (mode === 'usb') {
      const isCurrentUsb = videoDevices.find(d => d.deviceId === selectedDeviceId)?.label.toLowerCase().includes('usb');
      const usbDevice = videoDevices.find(d => d.label.toLowerCase().includes('usb'));
      
      if (usbDevice && !isCurrentUsb) {
        setSelectedDeviceId(usbDevice.deviceId);
        startStream(usbDevice.deviceId);
      } else if (selectedDeviceId) {
        startStream(selectedDeviceId);
      } else if (videoDevices.length > 0) {
        setSelectedDeviceId(videoDevices[0].deviceId);
        startStream(videoDevices[0].deviceId);
      } else {
        startStream('');
      }
    }
  };

  useEffect(() => {
    if (videoRef.current && stream && (sourceMode === 'camera' || sourceMode === 'usb')) {
      videoRef.current.srcObject = stream;
    }
  }, [stream, sourceMode, activeTab]);

  useEffect(() => {
    return () => stopStream();
  }, [stopStream]);

  // --- Close export menu on outside click ---
  useEffect(() => {
    if (!showExportMenu) return;
    const handler = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showExportMenu]);

  // --- Close sidebar on ESC key ---
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedRecord) setSelectedRecord(null);
        else if (sidebarOpen) setSidebarOpen(false);
        else if (showExportMenu) setShowExportMenu(false);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [selectedRecord, sidebarOpen, showExportMenu]);

  // --- Settings Logic ---
  const loadUsers = useCallback(async () => {
    try {
      const users = await usersApi.list();
      setUsersList(users);
    } catch (err: any) {
      addLog(`获取用户列表失败: ${err.message}`, 'error');
    }
  }, [addLog]);

  useEffect(() => {
    if (activeTab === 'settings') {
      loadUsers();
    }
  }, [activeTab, loadUsers]);

  const deleteUser = async (u: UserProfile) => {
    if (u.id === userId) {
      addLog("不能删除当前登录的用户", "error");
      return;
    }
    try {
      await usersApi.delete(u.id);
      loadUsers();
      addLog(`已删除用户: ${u.username}`, 'success');
    } catch (err: any) {
      addLog(`删除用户失败: ${err.message}`, 'error');
    }
  };

  // --- Load history from backend on auth ---
  const loadHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const records = await recordsApi.list();
      setHistory(records.map((r: any) => ({
        id: r.id,
        time: r.time,
        image: r.image || '',
        stats: { total: r.total_detections, fps: r.fps, avgConf: r.avg_confidence },
        detections: r.detections || [],
      })));
    } catch (err: any) {
      console.error('Failed to load history:', err);
      toast.error('加载历史记录失败');
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      loadHistory();
      // Check model status on backend
      inferenceApi.getStatus().then(status => {
        setModelLoaded(status.loaded);
        setModelName(status.model_name);
        setModelProvider(status.provider);
      }).catch(() => {});
    }
  }, [isAuthenticated, loadHistory]);

  // --- Auth Handlers ---
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setAuthError('请输入用户名和密码');
      return;
    }
    
    try {
      if (authMode === 'register') {
        await authApi.register(username, password);
        setAuthMode('login');
        setAuthError('');
        toast.success('注册成功，请登录');
        setPassword('');
      } else {
        const result = await authApi.login(username, password);
        setUserId(result.user_id || '');
        setIsAuthenticated(true);
        setAuthError('');
      }
    } catch (err: any) {
      setAuthError(err.message || '操作失败');
    }
  };

  const handleLogout = async () => {
    try { await authApi.logout(); } catch {}
    setIsAuthenticated(false);
    setUsername('');
    setPassword('');
    setUserId('');
    setModelLoaded(false);
    setMediaUrl(null);
    setHistory([]);
    setLogs([]);
    stopStream();
  };

  // --- File Handlers ---
  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    try {
      addLog(`正在上传模型到服务器: ${file.name}...`, 'info');
      toast.info(`正在上传模型: ${file.name}`);
      const result = await inferenceApi.uploadModel(file);
      setModelLoaded(true);
      setModelName(result.model_name);
      setModelProvider(result.provider);
      addLog(`模型加载成功！(${result.provider}) 输入节点: ${result.input_names.join(', ')}`, 'success');
      toast.success(`模型加载成功 (${result.provider})`);
    } catch (err: any) {
      console.error(err);
      addLog(`模型加载失败: ${err.message}`, 'error');
      toast.error(`模型加载失败: ${err.message}`);
    }
  };

  const handleMediaUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    
    const url = URL.createObjectURL(file);
    setMediaFile(file);
    setMediaUrl(url);
    
    if (file.type.startsWith('video/')) {
      setMediaType('video');
      addLog(`已加载视频: ${file.name}`, 'info');
    } else if (file.type.startsWith('image/')) {
      setMediaType('image');
      addLog(`已加载图片: ${file.name}`, 'info');
    } else {
      addLog(`不支持的文件类型: ${file.type}`, 'error');
      setMediaType(null);
    }
    
    // Reset canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setIsInferencing(false);
  };

  // --- Inference Logic (Backend CUDA) ---

  // Capture frame from video/image as base64
  const captureFrameAsBase64 = (source: HTMLVideoElement | HTMLImageElement): string => {
    const canvas = document.createElement('canvas');
    if (source instanceof HTMLVideoElement) {
      canvas.width = source.videoWidth;
      canvas.height = source.videoHeight;
    } else {
      canvas.width = source.naturalWidth;
      canvas.height = source.naturalHeight;
    }
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  const drawDetections = (detections: Detection[], ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.box;
      const width = x2 - x1;
      const height = y2 - y1;
      const color = COLORS[det.classId % COLORS.length];
      
      // Draw Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);
      
      // Draw Label Background
      const label = `${det.className} ${(det.score * 100).toFixed(1)}%`;
      ctx.font = '14px "Noto Sans SC", sans-serif';
      ctx.textBaseline = 'top';
      const textWidth = ctx.measureText(label).width;
      
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 24, textWidth + 10, 24);
      
      // Draw Label Text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, x1 + 5, y1 - 19);
    });
  };

  const saveRecordToBackend = async (snapshot: string, detections: Detection[], currentStats: { total: number; fps: number; avgConf: number }) => {
    const now = new Date();
    const time = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    try {
      const saved = await recordsApi.create({
        time,
        image: snapshot,
        total_detections: currentStats.total,
        fps: currentStats.fps,
        avg_confidence: currentStats.avgConf,
        detections,
      });
      setHistory(prev => [{
        id: saved.id,
        time,
        image: snapshot,
        stats: currentStats,
        detections,
      }, ...prev]);
    } catch {
      // Fallback: save locally if backend fails
      setHistory(prev => [{
        id: Math.random().toString(36).substr(2, 9),
        time,
        image: snapshot,
        stats: currentStats,
        detections,
      }, ...prev]);
    }
  };

  const detectFrame = async () => {
    if (!modelLoadedRef.current || !canvasRef.current) return;
    
    const source = mediaTypeRef.current === 'video' ? videoRef.current : imageRef.current;
    if (!source) return;

    // For video, check if it's playing
    if (mediaTypeRef.current === 'video' && (source as HTMLVideoElement).paused) {
      requestRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    // Throttle to target FPS
    const nowThrottle = performance.now();
    if (nowThrottle - lastTimeRef.current < 1000 / targetFpsRef.current) {
      requestRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    try {
      const imgWidth = mediaTypeRef.current === 'video' ? (source as HTMLVideoElement).videoWidth : (source as HTMLImageElement).naturalWidth;
      const imgHeight = mediaTypeRef.current === 'video' ? (source as HTMLVideoElement).videoHeight : (source as HTMLImageElement).naturalHeight;
      
      if (imgWidth === 0 || imgHeight === 0) {
        if (isInferencingRef.current) requestRef.current = requestAnimationFrame(detectFrame);
        return;
      }

      // Match canvas size to media size for accurate drawing
      if (canvasRef.current.width !== imgWidth) canvasRef.current.width = imgWidth;
      if (canvasRef.current.height !== imgHeight) canvasRef.current.height = imgHeight;

      // Capture frame and send to backend for CUDA inference
      const frameBase64 = captureFrameAsBase64(source);
      const result = await inferenceApi.detect(frameBase64, confThreshold, iouThreshold);
      
      const detections = result.detections;
      latestDetectionsRef.current = detections;
      
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) drawDetections(detections, ctx);

      if (mediaTypeRef.current === 'image' && imageRef.current) {
        const snapshot = captureSnapshot(imageRef.current, canvasRef.current);
        const currentStats = {
          total: detections.length,
          fps: 0,
          avgConf: detections.length > 0 ? detections.reduce((acc, d) => acc + d.score, 0) / detections.length : 0
        };
        setStats(currentStats);
        await saveRecordToBackend(snapshot, detections, currentStats);
        addLog("已自动保存图片识别记录", "success");
        setIsInferencing(false);
      } else {
        // Update Stats for video — rolling average FPS over last 10 frames
        const now = performance.now();
        const elapsed = now - lastTimeRef.current;
        if (lastTimeRef.current > 0 && elapsed > 0) {
          fpsHistoryRef.current.push(1000 / elapsed);
          if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
        }
        const avgFps = fpsHistoryRef.current.length > 0
          ? Math.round(fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length)
          : 0;
        lastMeasuredFpsRef.current = avgFps;
        const currentAvgConf = detections.length > 0 ? detections.reduce((acc, d) => acc + d.score, 0) / detections.length : 0;
        const currentStats = { total: detections.length, fps: avgFps, avgConf: currentAvgConf };
        setStats(currentStats);
        lastTimeRef.current = now;
        setDetectionHistory(prev => [...prev.slice(-29), detections.length]);

        // Submit annotated frame to backend buffer
        if (videoRef.current && canvasRef.current) {
          const snapshot = captureSnapshot(videoRef.current, canvasRef.current);
          const timestampMs = Math.round(now - videoSessionStartRef.current);
          inferenceApi.submitVideoFrame(
            videoSessionIdRef.current,
            snapshot,
            timestampMs,
            canvasRef.current.width,
            canvasRef.current.height,
          ).catch(() => {});
        }

        if (detections.length > 0) {
          const topDet = detections[0];
          addLog(`检测到 ${topDet.className} (${(topDet.score*100).toFixed(1)}%) [${result.inference_time_ms}ms]`, 'success');
        }
      }

    } catch (err: any) {
      console.error(err);
      addLog(`推理错误: ${err.message}`, 'error');
      setIsInferencing(false);
      isInferencingRef.current = false;
      return;
    }

    if (isInferencingRef.current && mediaTypeRef.current === 'video') {
      requestRef.current = requestAnimationFrame(detectFrame);
    }
  };

  useEffect(() => {
    if (isInferencing) {
      if (mediaTypeRef.current === 'video' && videoRef.current) {
        videoRef.current.play();
      }
      lastTimeRef.current = 0;
      fpsHistoryRef.current = [];
      lastMeasuredFpsRef.current = 0;
      if (mediaTypeRef.current === 'video') {
        videoSessionIdRef.current = crypto.randomUUID();
        videoSessionStartRef.current = performance.now();
      }
      detectFrame();
    } else {
      if (mediaTypeRef.current === 'video' && videoRef.current && sourceModeRef.current === 'file') {
        videoRef.current.pause();
      }
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (mediaTypeRef.current === 'video' && videoSessionIdRef.current) {
        const sessionId = videoSessionIdRef.current;
        videoSessionIdRef.current = '';
        inferenceApi.finalizeVideo(sessionId, targetFpsRef.current, 10).then(async result => {
          if (!result.success || result.clips.length === 0) return;
          const now = new Date();
          const time = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
          try {
            await recordsApi.create({
              time,
              image: '',
              total_detections: latestDetectionsRef.current.length,
              fps: targetFpsRef.current,
              avg_confidence: latestDetectionsRef.current.length > 0
                ? latestDetectionsRef.current.reduce((a, d) => a + d.score, 0) / latestDetectionsRef.current.length
                : 0,
              detections: latestDetectionsRef.current,
              video_clips: result.clips,
            });
            addLog(`视频识别完成，已保存 ${result.clips.length} 个视频片段`, 'success');
            toast.success(`视频已保存，共 ${result.clips.length} 个片段`);
          } catch {
            addLog('视频记录保存失败', 'error');
          }
        }).catch(() => {
          addLog('视频编码失败', 'error');
        });
      }
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isInferencing]);

  const toggleInference = () => {
    if (!modelLoaded) {
      addLog("请先加载 ONNX 模型。", "warning");
      return;
    }
    if (sourceMode === 'file' && !mediaUrl) {
      addLog("请先加载图片或视频。", "warning");
      return;
    }
    if ((sourceMode === 'camera' || sourceMode === 'usb') && !stream) {
      addLog("请先开启视频设备。", "warning");
      return;
    }

    setIsInferencing(!isInferencing);
  };

  // --- Delete Record ---
  const deleteRecord = async (record: HistoryRecord, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    try {
      await recordsApi.delete(record.id);
      setHistory(prev => prev.filter(r => r.id !== record.id));
      if (selectedRecord?.id === record.id) setSelectedRecord(null);
      toast.success('记录已删除');
      addLog('已删除一条识别记录', 'info');
    } catch (err: any) {
      toast.error(`删除失败: ${err.message}`);
    }
  };

  // --- Export Functions ---
  const exportToCSV = () => {
    if (history.length === 0) { toast.warning('暂无数据可导出'); return; }
    const headers = ['时间', '目标数', 'FPS', '平均置信度', '检测类别详情'];
    const rows = history.map(r => [
      r.time,
      r.stats.total,
      r.stats.fps,
      (r.stats.avgConf * 100).toFixed(1) + '%',
      r.detections.map(d => `${d.className}(${(d.score * 100).toFixed(0)}%)`).join('; '),
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `YOLO_检测记录_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
    toast.success('CSV 文件已导出');
    setShowExportMenu(false);
  };

  const exportToJSON = () => {
    if (history.length === 0) { toast.warning('暂无数据可导出'); return; }
    const data = history.map(r => ({
      time: r.time,
      total_detections: r.stats.total,
      fps: r.stats.fps,
      avg_confidence: r.stats.avgConf,
      detections: r.detections,
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `YOLO_检测记录_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    toast.success('JSON 文件已导出');
    setShowExportMenu(false);
  };

  // --- Batch Detection ---
  const handleBatchUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    if (!modelLoaded) { toast.warning('请先加载模型'); return; }

    const imageFiles = (Array.from(files) as File[]).filter(f => f.type.startsWith('image/'));
    if (imageFiles.length === 0) { toast.warning('未选择有效的图片文件'); return; }

    setBatchFiles(imageFiles);
    setBatchProgress({ current: 0, total: imageFiles.length });
    toast.info(`开始批量检测 ${imageFiles.length} 张图片...`);

    for (let i = 0; i < imageFiles.length; i++) {
      setBatchProgress({ current: i + 1, total: imageFiles.length });
      try {
        const file = imageFiles[i];
        const url = URL.createObjectURL(file);
        const img = new window.Image();
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error('图片加载失败'));
          img.src = url;
        });

        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        const base64 = canvas.toDataURL('image/jpeg', 0.85);

        const result = await inferenceApi.detect(base64, confThreshold, iouThreshold);

        // Draw detections on snapshot
        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.width = img.naturalWidth;
        overlayCanvas.height = img.naturalHeight;
        const octx = overlayCanvas.getContext('2d')!;
        result.detections.forEach(det => {
          const [x1, y1, x2, y2] = det.box;
          octx.strokeStyle = COLORS[det.classId % COLORS.length];
          octx.lineWidth = 3;
          octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          const label = `${det.className} ${(det.score * 100).toFixed(1)}%`;
          octx.font = '14px sans-serif';
          const tw = octx.measureText(label).width;
          octx.fillStyle = COLORS[det.classId % COLORS.length];
          octx.fillRect(x1, y1 - 24, tw + 10, 24);
          octx.fillStyle = '#000';
          octx.fillText(label, x1 + 5, y1 - 7);
        });

        const snapCanvas = document.createElement('canvas');
        snapCanvas.width = img.naturalWidth;
        snapCanvas.height = img.naturalHeight;
        const sctx = snapCanvas.getContext('2d')!;
        sctx.drawImage(img, 0, 0);
        sctx.drawImage(overlayCanvas, 0, 0);
        const snapshot = snapCanvas.toDataURL('image/jpeg', 0.8);

        const currentStats = {
          total: result.detections.length,
          fps: 0,
          avgConf: result.detections.length > 0
            ? result.detections.reduce((acc, d) => acc + d.score, 0) / result.detections.length
            : 0,
        };
        await saveRecordToBackend(snapshot, result.detections, currentStats);
        addLog(`[批量 ${i + 1}/${imageFiles.length}] ${file.name}: 检测到 ${result.detections.length} 个目标`, 'success');
        URL.revokeObjectURL(url);
      } catch (err: any) {
        addLog(`[批量 ${i + 1}/${imageFiles.length}] 失败: ${err.message}`, 'error');
      }
    }

    setBatchProgress(null);
    setBatchFiles([]);
    toast.success(`批量检测完成！已处理 ${imageFiles.length} 张图片`);
  };

  const classStats = useMemo(() => {
    const statsMap: Record<string, { count: number, totalConf: number }> = {};
    history.forEach(record => {
      record.detections.forEach(det => {
        if (!statsMap[det.className]) {
          statsMap[det.className] = { count: 0, totalConf: 0 };
        }
        statsMap[det.className].count += 1;
        statsMap[det.className].totalConf += det.score;
      });
    });

    return Object.entries(statsMap).map(([className, data]) => ({
      name: className,
      count: data.count,
      avgConf: Number(((data.totalConf / data.count) * 100).toFixed(1))
    })).sort((a, b) => b.count - a.count);
  }, [history]);

  if (!isAuthenticated) {
    return (
      <div className="flex h-screen bg-[#0a0d12] items-center justify-center font-sans selection:bg-[#00daf3]/30">
        <div className="w-full max-w-md p-8 bg-[#10141a] border border-white/10 rounded-2xl shadow-2xl shadow-[#00daf3]/5 scale-in">
          <div className="flex flex-col items-center mb-8">
            <div className="w-12 h-12 bg-[#00daf3]/20 rounded-xl flex items-center justify-center border border-[#00daf3]/30 mb-4">
              <Box className="text-[#00daf3] w-6 h-6" />
            </div>
            <h1 className="text-2xl font-bold text-white tracking-widest">综合观测系统</h1>
            <p className="text-xs text-[#00daf3] uppercase tracking-[0.3em] mt-2">YOLO Vision Engine</p>
          </div>

          <form onSubmit={handleAuth} className="space-y-5">
            {authError && (
              <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-500 text-sm">
                <AlertCircle className="w-4 h-4" />
                {authError}
              </div>
            )}
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">用户名</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input 
                  type="text" 
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  className="w-full bg-[#181c22] border border-white/5 rounded-lg py-3 pl-10 pr-4 text-white focus:outline-none focus:border-[#00daf3]/50 transition-colors"
                  placeholder="输入用户名"
                />
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">密码</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input 
                  type="password" 
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full bg-[#181c22] border border-white/5 rounded-lg py-3 pl-10 pr-4 text-white focus:outline-none focus:border-[#00daf3]/50 transition-colors"
                  placeholder="输入密码"
                />
              </div>
            </div>
            <button 
              type="submit"
              className="w-full py-3 bg-[#00daf3] text-[#001f24] font-bold rounded-lg uppercase tracking-widest hover:bg-[#00daf3]/90 transition-colors shadow-[0_0_20px_rgba(0,218,243,0.2)]"
            >
              {authMode === 'login' ? '登 录' : '注 册'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <button 
              onClick={() => {
                setAuthMode(authMode === 'login' ? 'register' : 'login');
                setAuthError('');
              }}
              className="text-xs text-slate-400 hover:text-[#00daf3] transition-colors"
            >
              {authMode === 'login' ? '没有账号？点击注册' : '已有账号？点击登录'}
            </button>
          </div>
        </div>
        <ToastContainer />
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#10141a] text-[#dfe2eb] font-sans overflow-hidden selection:bg-[#00daf3]/30">
      
      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed lg:relative inset-y-0 left-0 w-64 border-r border-white/5 bg-[#10141a] shadow-2xl flex flex-col z-40 transition-transform duration-300 ease-out ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0`}>
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-[#00daf3]/20 rounded-lg flex items-center justify-center border border-[#00daf3]/30">
              <Box className="text-[#00daf3] w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-[#00daf3] tracking-tighter">YOLOv11</h1>
              <p className="text-[10px] text-slate-500 uppercase tracking-[0.2em]">视觉引擎</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 py-4 space-y-1">
          <div 
            onClick={() => { setActiveTab('dashboard'); setSidebarOpen(false); }}
            className={`sidebar-nav-item flex items-center gap-3 px-6 py-3 transition-all cursor-pointer ${activeTab === 'dashboard' ? 'bg-[#00daf3]/10 text-[#00daf3] border-r-2 border-[#00daf3]' : 'text-slate-400 hover:text-[#00daf3] hover:bg-white/5'}`}
          >
            <LayoutDashboard className="w-5 h-5" />
            <span className="text-sm font-medium tracking-wider">控制台</span>
          </div>
          <div 
            onClick={() => { setActiveTab('history'); setSidebarOpen(false); }}
            className={`sidebar-nav-item flex items-center gap-3 px-6 py-3 transition-all cursor-pointer ${activeTab === 'history' ? 'bg-[#00daf3]/10 text-[#00daf3] border-r-2 border-[#00daf3]' : 'text-slate-400 hover:text-[#00daf3] hover:bg-white/5'}`}
          >
            <History className="w-5 h-5" />
            <span className="text-sm font-medium tracking-wider">历史记录</span>
          </div>
          <div 
            onClick={() => { setActiveTab('analytics'); setSidebarOpen(false); }}
            className={`sidebar-nav-item flex items-center gap-3 px-6 py-3 transition-all cursor-pointer ${activeTab === 'analytics' ? 'bg-[#00daf3]/10 text-[#00daf3] border-r-2 border-[#00daf3]' : 'text-slate-400 hover:text-[#00daf3] hover:bg-white/5'}`}
          >
            <BarChart2 className="w-5 h-5" />
            <span className="text-sm font-medium tracking-wider">数据分析</span>
          </div>
          <div 
            onClick={() => { setActiveTab('settings'); setSidebarOpen(false); }}
            className={`sidebar-nav-item flex items-center gap-3 px-6 py-3 transition-all cursor-pointer ${activeTab === 'settings' ? 'bg-[#00daf3]/10 text-[#00daf3] border-r-2 border-[#00daf3]' : 'text-slate-400 hover:text-[#00daf3] hover:bg-white/5'}`}
          >
            <Settings className="w-5 h-5" />
            <span className="text-sm font-medium tracking-wider">系统设置</span>
          </div>
        </nav>
        <div className="p-6 border-t border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-600 flex items-center justify-center">
              <User className="w-4 h-4 text-slate-400" />
            </div>
            <div>
              <p className="text-xs font-medium text-white">{username}</p>
              <p className="text-[10px] text-[#00daf3]">操作员权限</p>
            </div>
          </div>
          <button 
            onClick={handleLogout} 
            className="mt-4 w-full flex items-center justify-center gap-2 py-2 bg-red-500/10 text-red-500 rounded hover:bg-red-500/20 transition-colors text-xs font-bold tracking-widest"
          >
            <LogOut className="w-3 h-3" />
            退出登录
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-16 border-b border-white/5 bg-[#10141a]/80 backdrop-blur-xl flex items-center justify-between px-8 z-10">
          <div className="flex items-center gap-4 lg:gap-6">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="lg:hidden text-slate-400 hover:text-[#00daf3] transition-colors p-1">
              <Menu className="w-6 h-6" />
            </button>
            <span className="text-base lg:text-lg font-black text-white uppercase tracking-widest">综合观测系统</span>
            <div className="h-4 w-px bg-white/10 hidden lg:block"></div>
            <span className="text-[10px] lg:text-xs font-medium text-[#00daf3] bg-[#00daf3]/10 px-2 py-1 rounded border border-[#00daf3]/20 hidden sm:inline-flex">
              {modelLoaded ? `引擎就绪 (${modelProvider})` : '等待加载模型'}
            </span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <Cpu className="w-5 h-5 hover:text-[#00daf3] cursor-pointer transition-colors" />
            <Activity className="w-5 h-5 hover:text-[#00daf3] cursor-pointer transition-colors" />
          </div>
        </header>

        {/* Content Area */}
        {activeTab === 'dashboard' ? (
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          
          {/* Left: Viewer & Controls */}
          <div className="flex-1 flex flex-col p-6 gap-6 overflow-y-auto custom-scrollbar">
            
            {/* Viewer */}
            <div className="relative flex-1 min-h-[400px] bg-[#181c22] rounded-xl border border-white/5 overflow-hidden flex items-center justify-center shadow-2xl shadow-cyan-900/5 group">
              {!mediaUrl ? (
                <div className="text-center text-slate-500 flex flex-col items-center gap-3">
                  <Video className="w-12 h-12 opacity-50" />
                  <p className="text-sm uppercase tracking-widest font-medium">暂无媒体源</p>
                </div>
              ) : (
                <>
                  {mediaType === 'video' ? (
                    <video
                      ref={videoRef}
                      src={sourceMode === 'file' ? mediaUrl : undefined}
                      className="w-full h-full object-contain"
                      muted
                      playsInline
                      autoPlay={sourceMode === 'camera' || sourceMode === 'usb'}
                      onEnded={() => { if (isInferencing) setIsInferencing(false); }}
                    />
                  ) : (
                    <img 
                      ref={imageRef} 
                      src={mediaUrl} 
                      className="w-full h-full object-contain"
                      alt="Source"
                    />
                  )}
                  <canvas 
                    ref={canvasRef} 
                    className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none"
                  />
                </>
              )}
              
              {/* Overlay Status */}
              <div className="absolute top-4 left-4 flex gap-2">
                <span className="bg-[#10141a]/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-bold text-[#00daf3] border border-white/5 uppercase tracking-wider">
                  {mediaType === 'video' ? '视频' : mediaType === 'image' ? '图片' : '空闲'}
                </span>
                {isInferencing && (
                  <span className="bg-[#00daf3]/20 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-bold text-[#00daf3] border border-[#00daf3]/30 uppercase tracking-wider flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#00daf3] animate-pulse"></span>
                    推理中
                  </span>
                )}
              </div>
            </div>

            {/* Controls */}
            <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
              
              {/* Media Source */}
              <div className="space-y-3 col-span-1 md:col-span-2">
                <div className="flex justify-between items-center">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">媒体源</label>
                  <div className="flex bg-[#10141a] rounded-lg p-1 border border-white/5">
                    <button onClick={() => switchToMode('file')} className={`px-3 py-1 text-[10px] rounded transition-colors ${sourceMode === 'file' ? 'bg-[#00daf3]/20 text-[#00daf3]' : 'text-slate-500 hover:text-white'}`}>手动上传</button>
                    <button onClick={() => switchToMode('camera')} className={`px-3 py-1 text-[10px] rounded transition-colors ${sourceMode === 'camera' ? 'bg-[#00daf3]/20 text-[#00daf3]' : 'text-slate-500 hover:text-white'}`}>摄像头</button>
                    <button onClick={() => switchToMode('usb')} className={`px-3 py-1 text-[10px] rounded transition-colors ${sourceMode === 'usb' ? 'bg-[#00daf3]/20 text-[#00daf3]' : 'text-slate-500 hover:text-white'}`}>USB输入</button>
                  </div>
                </div>
                
                {sourceMode === 'file' && (
                  <label className="flex items-center gap-3 p-3 bg-[#10141a] border border-white/5 rounded-lg cursor-pointer hover:border-[#00daf3]/50 transition-colors group">
                    <div className="w-8 h-8 bg-purple-500/10 rounded flex items-center justify-center text-purple-400">
                      <ImageIcon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <p className="text-xs font-medium text-white truncate">{mediaFile ? mediaFile.name : '暂无媒体'}</p>
                      <p className="text-[10px] text-slate-500">上传图片/视频</p>
                    </div>
                    <input type="file" accept="image/*,video/*" onChange={handleMediaUpload} className="hidden" />
                  </label>
                )}

                {sourceMode === 'camera' && (
                  <div className="flex items-center gap-3 p-3 bg-[#10141a] border border-white/5 rounded-lg">
                    <div className="w-8 h-8 bg-emerald-500/10 rounded flex items-center justify-center text-emerald-400">
                      <Camera className="w-4 h-4" />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <p className="text-xs font-medium text-white truncate">内置摄像头</p>
                      <p className="text-[10px] text-slate-500">正在使用默认摄像头</p>
                    </div>
                  </div>
                )}

                {sourceMode === 'usb' && (
                  <div className="flex items-center gap-3 p-3 bg-[#10141a] border border-white/5 rounded-lg">
                    <div className="w-8 h-8 bg-blue-500/10 rounded flex items-center justify-center text-blue-400">
                      <Usb className="w-4 h-4" />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <select 
                        className="w-full bg-transparent text-xs text-white outline-none appearance-none cursor-pointer"
                        value={selectedDeviceId}
                        onChange={(e) => {
                          setSelectedDeviceId(e.target.value);
                          startStream(e.target.value);
                        }}
                      >
                        {videoDevices.length === 0 && <option value="">未检测到设备</option>}
                        {videoDevices.map(d => (
                          <option key={d.deviceId} value={d.deviceId} className="bg-[#10141a]">
                            {d.label || `视频设备 ${d.deviceId.substring(0,5)}...`}
                          </option>
                        ))}
                      </select>
                      <p className="text-[10px] text-slate-500">选择USB视频输入设备</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Thresholds */}
              <div className="space-y-4 flex flex-col justify-center">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-[10px] uppercase text-slate-500 font-bold tracking-widest">置信度</span>
                    <span className="text-[#00daf3] font-mono text-xs">{confThreshold.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" min="0.1" max="0.95" step="0.05" 
                    value={confThreshold} onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#00daf3]" 
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-[10px] uppercase text-slate-500 font-bold tracking-widest">交并比 (IOU)</span>
                    <span className="text-[#00daf3] font-mono text-xs">{iouThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0.1" max="0.95" step="0.05"
                    value={iouThreshold} onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#00daf3]"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-[10px] uppercase text-slate-500 font-bold tracking-widest">目标 FPS</span>
                    <span className="text-[#00daf3] font-mono text-xs">{targetFps} FPS</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => {
                        const opts = [15, 30, 60, 90, 150, 200, 300];
                        const idx = opts.indexOf(targetFps);
                        if (idx > 0) setTargetFps(opts[idx - 1]);
                      }}
                      className="w-6 h-6 flex items-center justify-center rounded bg-white/5 hover:bg-white/10 text-slate-300 text-sm font-bold disabled:opacity-30 transition-colors"
                      disabled={targetFps === 15}
                    >‹</button>
                    <div className="flex-1 flex justify-center gap-1">
                      {[15, 30, 60, 90, 150, 200, 300].map(fps => (
                        <button
                          key={fps}
                          onClick={() => setTargetFps(fps)}
                          className={`flex-1 py-1 rounded text-[10px] font-mono transition-colors ${
                            fps === targetFps
                              ? 'bg-[#00daf3]/20 text-[#00daf3] border border-[#00daf3]/40'
                              : 'bg-white/5 text-slate-500 hover:text-slate-300 hover:bg-white/10'
                          }`}
                        >{fps}</button>
                      ))}
                    </div>
                    <button
                      onClick={() => {
                        const opts = [15, 30, 60, 90, 150, 200, 300];
                        const idx = opts.indexOf(targetFps);
                        if (idx < opts.length - 1) setTargetFps(opts[idx + 1]);
                      }}
                      className="w-6 h-6 flex items-center justify-center rounded bg-white/5 hover:bg-white/10 text-slate-300 text-sm font-bold disabled:opacity-30 transition-colors"
                      disabled={targetFps === 300}
                    >›</button>
                  </div>
                </div>
              </div>

              {/* Action */}
              <div className="flex flex-col gap-3 justify-end">
                <button 
                  onClick={toggleInference}
                  disabled={!modelLoaded || !mediaUrl}
                  className={`glow-btn w-full py-4 rounded-lg font-bold text-sm uppercase tracking-widest transition-all flex items-center justify-center gap-2
                    ${!modelLoaded || !mediaUrl 
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                      : isInferencing 
                        ? 'bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20' 
                        : 'bg-[#00daf3] text-[#001f24] hover:bg-[#00daf3]/90 shadow-[0_0_20px_rgba(0,218,243,0.3)]'
                    }`}
                >
                  {isInferencing ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
                  {isInferencing ? '停止引擎' : '启动引擎'}
                </button>

                {/* Batch Upload */}
                <label className="w-full py-2.5 rounded-lg font-bold text-[10px] uppercase tracking-widest transition-all flex items-center justify-center gap-2 cursor-pointer bg-[#10141a] border border-white/10 text-slate-400 hover:text-[#00daf3] hover:border-[#00daf3]/30">
                  <Images className="w-4 h-4" />
                  {batchProgress ? `批量检测中 ${batchProgress.current}/${batchProgress.total}` : '批量检测'}
                  <input type="file" accept="image/*" multiple onChange={handleBatchUpload} className="hidden" disabled={!!batchProgress} />
                </label>
                {batchProgress && (
                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-[#00daf3] transition-all duration-300 rounded-full" style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }} />
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* Right: Stats & Logs */}
          <aside className="w-full lg:w-80 bg-[#181c22] border-l border-white/5 flex flex-col">
            {/* Stats */}
            <div className="p-6 border-b border-white/5 space-y-4">
              <h3 className="text-xs font-bold text-white uppercase tracking-widest flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#00daf3]" />
                遥测数据
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-[#10141a] p-4 rounded-lg border border-white/5">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">帧率 (FPS)</p>
                  <p className="text-2xl font-bold text-white">{stats.fps}</p>
                </div>
                <div className="bg-[#10141a] p-4 rounded-lg border border-white/5">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">目标数</p>
                  <p className="text-2xl font-bold text-[#00daf3]">{stats.total}</p>
                </div>
                <div className="col-span-2 bg-[#10141a] p-4 rounded-lg border border-white/5">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">平均置信度</p>
                  <div className="flex items-center gap-3">
                    <p className="text-xl font-bold text-white">{(stats.avgConf * 100).toFixed(1)}%</p>
                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-[#00daf3] to-[#009fb2] rounded-full transition-all duration-500" style={{ width: `${stats.avgConf * 100}%` }}></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Real-time Sparkline */}
              {detectionHistory.length > 1 && (
                <div className="bg-[#10141a] p-4 rounded-lg border border-white/5">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">检测趋势</p>
                  <svg viewBox="0 0 200 40" className="w-full h-10" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="0" y1="20" x2="200" y2="20" stroke="#ffffff08" strokeWidth="0.5" />
                    {/* Area fill */}
                    <path
                      d={(() => {
                        const max = Math.max(...detectionHistory, 1);
                        const points = detectionHistory.map((v, i) => {
                          const x = (i / (detectionHistory.length - 1)) * 200;
                          const y = 38 - (v / max) * 36;
                          return `${x},${y}`;
                        });
                        return `M0,38 L${points.join(' L')} L200,38 Z`;
                      })()}
                      fill="url(#sparkGradient)"
                    />
                    {/* Line */}
                    <polyline
                      points={detectionHistory.map((v, i) => {
                        const max = Math.max(...detectionHistory, 1);
                        const x = (i / (detectionHistory.length - 1)) * 200;
                        const y = 38 - (v / max) * 36;
                        return `${x},${y}`;
                      }).join(' ')}
                      fill="none"
                      stroke="#00daf3"
                      strokeWidth="1.5"
                      strokeLinejoin="round"
                    />
                    {/* Current dot */}
                    {detectionHistory.length > 0 && (() => {
                      const max = Math.max(...detectionHistory, 1);
                      const lastIdx = detectionHistory.length - 1;
                      const x = (lastIdx / (detectionHistory.length - 1)) * 200;
                      const y = 38 - (detectionHistory[lastIdx] / max) * 36;
                      return <circle cx={x} cy={y} r="2.5" fill="#00daf3" className="animate-pulse" />;
                    })()}
                    <defs>
                      <linearGradient id="sparkGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#00daf3" stopOpacity="0.3" />
                        <stop offset="100%" stopColor="#00daf3" stopOpacity="0.02" />
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
              )}
            </div>

            {/* Logs */}
            <div className="flex-1 flex flex-col p-6 overflow-hidden">
              <h3 className="text-xs font-bold text-white uppercase tracking-widest flex items-center gap-2 mb-4">
                <Terminal className="w-4 h-4 text-[#00daf3]" />
                系统日志
              </h3>
              <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 pr-2">
                {logs.length === 0 ? (
                  <p className="text-xs text-slate-600 font-mono">系统已初始化。等待输入...</p>
                ) : (
                  logs.map(log => (
                    <div key={log.id} className="bg-[#10141a] p-3 rounded-lg border border-white/5 flex gap-3 group">
                      <div className={`w-1 rounded-full ${
                        log.type === 'error' ? 'bg-red-500' : 
                        log.type === 'success' ? 'bg-green-500' : 
                        log.type === 'warning' ? 'bg-yellow-500' : 'bg-[#00daf3]'
                      }`}></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-[9px] text-slate-500 font-mono">{log.time}</span>
                          {log.type === 'error' && <AlertCircle className="w-3 h-3 text-red-500" />}
                          {log.type === 'success' && <CheckCircle2 className="w-3 h-3 text-green-500" />}
                        </div>
                        <p className="text-[11px] text-slate-300 break-words leading-relaxed">{log.message}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </aside>

        </div>
        ) : activeTab === 'history' ? (
          <div className="flex-1 p-6 lg:p-8 overflow-y-auto custom-scrollbar page-enter">
            {/* Header with Export */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
              <h2 className="text-xl lg:text-2xl font-bold text-white flex items-center gap-3">
                <History className="w-6 h-6 text-[#00daf3]" />
                识别历史记录
                {history.length > 0 && (
                  <span className="text-xs font-normal text-slate-500 bg-white/5 px-2 py-0.5 rounded-full">{history.length} 条</span>
                )}
              </h2>
              {history.length > 0 && (
                <div className="relative" ref={exportMenuRef}>
                  <button
                    onClick={() => setShowExportMenu(!showExportMenu)}
                    className="flex items-center gap-2 px-4 py-2 bg-[#181c22] border border-white/10 rounded-lg text-sm text-slate-300 hover:text-[#00daf3] hover:border-[#00daf3]/30 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    导出数据
                    <ChevronDown className={`w-3 h-3 transition-transform ${showExportMenu ? 'rotate-180' : ''}`} />
                  </button>
                  {showExportMenu && (
                    <div className="export-dropdown dropdown-enter bg-[#1a1f27] border border-white/10 rounded-xl shadow-2xl overflow-hidden">
                      <button
                        onClick={exportToCSV}
                        className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-300 hover:bg-[#00daf3]/10 hover:text-[#00daf3] transition-colors"
                      >
                        <FileSpreadsheet className="w-4 h-4" />
                        导出为 CSV
                      </button>
                      <div className="border-t border-white/5" />
                      <button
                        onClick={exportToJSON}
                        className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-300 hover:bg-[#00daf3]/10 hover:text-[#00daf3] transition-colors"
                      >
                        <FileJson2 className="w-4 h-4" />
                        导出为 JSON
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Skeleton Loading */}
            {isLoadingHistory ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="bg-[#181c22] border border-white/5 rounded-xl overflow-hidden">
                    <div className="h-48 bg-slate-800/50 skeleton-pulse" />
                    <div className="p-4 bg-[#10141a] border-t border-white/5 space-y-3">
                      <div className="grid grid-cols-3 gap-2">
                        <div className="h-8 bg-slate-800/50 rounded skeleton-pulse" />
                        <div className="h-8 bg-slate-800/50 rounded skeleton-pulse" style={{ animationDelay: '0.1s' }} />
                        <div className="h-8 bg-slate-800/50 rounded skeleton-pulse" style={{ animationDelay: '0.2s' }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : history.length === 0 ? (
              <div className="text-center text-slate-500 py-20 flex flex-col items-center gap-4 scale-in">
                <History className="w-16 h-16 opacity-20" />
                <p className="text-sm">暂无历史记录。请先在控制台运行识别引擎。</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {history.map((record, index) => (
                  <div 
                    key={record.id} 
                    onClick={() => setSelectedRecord(record)}
                    className="history-card bg-[#181c22] border border-white/5 rounded-xl overflow-hidden shadow-lg group hover:border-[#00daf3]/30 cursor-pointer card-enter"
                    style={{ animationDelay: `${index * 0.05}s` }}
                  >
                    <div className="h-48 bg-black relative">
                      {record.image
                        ? <img src={record.image} alt="Snapshot" className="w-full h-full object-contain" />
                        : <div className="w-full h-full flex flex-col items-center justify-center text-white/30 gap-2">
                            <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 10l4.553-2.069A1 1 0 0121 8.82V15.18a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" /></svg>
                            <span className="text-xs">视频记录</span>
                          </div>
                      }
                      <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-[10px] text-white font-mono border border-white/10">
                        {record.time}
                      </div>
                      {/* Delete Button */}
                      <button
                        onClick={(e) => deleteRecord(record, e)}
                        className="card-delete-btn absolute top-2 left-2 bg-red-500/80 hover:bg-red-500 backdrop-blur-sm p-1.5 rounded-lg transition-colors"
                        title="删除记录"
                      >
                        <Trash2 className="w-3.5 h-3.5 text-white" />
                      </button>
                    </div>
                    <div className="p-4 bg-[#10141a] border-t border-white/5">
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div>
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">目标数</p>
                          <p className="text-lg font-bold text-[#00daf3]">{record.stats.total}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">FPS</p>
                          <p className="text-lg font-bold text-white">{record.stats.fps}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">平均置信度</p>
                          <p className="text-lg font-bold text-white">{(record.stats.avgConf * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : activeTab === 'analytics' ? (
          <div className="flex-1 p-6 lg:p-8 overflow-y-auto custom-scrollbar page-enter">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <BarChart2 className="w-6 h-6 text-[#00daf3]" />
              数据分析与统计
            </h2>
            {history.length === 0 ? (
              <div className="text-center text-slate-500 py-20 flex flex-col items-center gap-4">
                <BarChart2 className="w-16 h-16 opacity-20" />
                <p>暂无数据。请先在控制台运行识别引擎以生成数据。</p>
              </div>
            ) : classStats.length === 0 ? (
              <div className="text-center text-slate-500 py-20 flex flex-col items-center gap-4">
                <Box className="w-16 h-16 opacity-20" />
                <p>历史记录中未检测到任何目标。</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Chart 1: Counts */}
                <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 shadow-lg">
                  <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                    <Box className="w-4 h-4 text-[#00daf3]" />
                    各类别识别总数
                  </h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={classStats} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="name" stroke="#888" tick={{fill: '#888', fontSize: 12}} />
                        <YAxis stroke="#888" tick={{fill: '#888', fontSize: 12}} allowDecimals={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#10141a', borderColor: '#333', color: '#fff', borderRadius: '8px' }}
                          itemStyle={{ color: '#00daf3' }}
                          cursor={{fill: '#ffffff0a'}}
                        />
                        <Bar dataKey="count" name="识别数量" fill="#00daf3" radius={[4, 4, 0, 0]}>
                          {classStats.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 2: Average Confidence */}
                <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 shadow-lg">
                  <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#f06600]" />
                    各类别平均置信度 (%)
                  </h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={classStats} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="name" stroke="#888" tick={{fill: '#888', fontSize: 12}} />
                        <YAxis stroke="#888" tick={{fill: '#888', fontSize: 12}} domain={[0, 100]} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#10141a', borderColor: '#333', color: '#fff', borderRadius: '8px' }}
                          itemStyle={{ color: '#f06600' }}
                          cursor={{fill: '#ffffff0a'}}
                        />
                        <Bar dataKey="avgConf" name="平均置信度" fill="#f06600" radius={[4, 4, 0, 0]}>
                          {classStats.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[(index + 2) % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* Chart 3: Composition Pie Chart */}
                <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 shadow-lg col-span-1 lg:col-span-2">
                  <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                    <PieChartIcon className="w-4 h-4 text-purple-400" />
                    目标类别占比
                  </h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={classStats}
                          cx="50%"
                          cy="50%"
                          innerRadius={80}
                          outerRadius={120}
                          paddingAngle={5}
                          dataKey="count"
                          nameKey="name"
                          label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                          labelLine={{ stroke: '#888' }}
                        >
                          {classStats.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#10141a', borderColor: '#333', color: '#fff', borderRadius: '8px' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Legend verticalAlign="bottom" height={36} wrapperStyle={{ color: '#888' }}/>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : activeTab === 'settings' ? (
          <div className="flex-1 p-6 lg:p-8 overflow-y-auto custom-scrollbar page-enter">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Settings className="w-6 h-6 text-[#00daf3]" />
              系统设置
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Model Configuration */}
              <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 shadow-lg">
                <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-[#00daf3]" />
                  模型配置
                </h3>
                <div className="space-y-4">
                  <label className="flex items-center gap-3 p-4 bg-[#10141a] border border-white/5 rounded-lg cursor-pointer hover:border-[#00daf3]/50 transition-colors group">
                    <div className="w-10 h-10 bg-[#00daf3]/10 rounded-lg flex items-center justify-center text-[#00daf3] group-hover:scale-110 transition-transform">
                      <Upload className="w-5 h-5" />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <p className="text-sm font-bold text-white truncate">{modelName !== '未加载模型' ? modelName : '上传 ONNX / PT 模型'}</p>
                      <p className="text-xs text-slate-500 mt-1">支持 YOLOv11 格式</p>
                    </div>
                    <input type="file" accept=".onnx,.pt" onChange={handleModelUpload} className="hidden" />
                  </label>
                  {modelName !== '未加载模型' && (
                    <div className="space-y-2">
                      <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                        <span className="text-xs text-green-400">模型已加载并准备就绪</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-3 bg-[#10141a] border border-white/5 rounded-lg">
                          <p className="text-[10px] text-slate-500 uppercase tracking-widest">推理引擎</p>
                          <p className="text-sm font-bold text-[#00daf3] mt-1">{modelProvider || 'N/A'}</p>
                        </div>
                        <div className="p-3 bg-[#10141a] border border-white/5 rounded-lg">
                          <p className="text-[10px] text-slate-500 uppercase tracking-widest">模型名称</p>
                          <p className="text-sm font-bold text-white mt-1 truncate">{modelName}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* User Management */}
              <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 shadow-lg">
                <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                  <Users className="w-4 h-4 text-purple-400" />
                  用户管理
                  {usersList.length > 0 && (
                    <span className="text-[10px] text-slate-500 bg-white/5 px-2 py-0.5 rounded-full font-normal">{usersList.length} 位</span>
                  )}
                </h3>
                <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
                  {usersList.length === 0 ? (
                    <p className="text-xs text-slate-500 text-center py-4">暂无其他用户</p>
                  ) : (
                    usersList.map((user) => (
                      <div key={user.id} className="flex items-center justify-between p-3 bg-[#10141a] border border-white/5 rounded-lg hover:border-white/10 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${user.username === username ? 'bg-[#00daf3]/20 border border-[#00daf3]/30' : 'bg-slate-800'}`}>
                            <User className={`w-4 h-4 ${user.username === username ? 'text-[#00daf3]' : 'text-slate-400'}`} />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-white">{user.username}</p>
                            <p className="text-[10px] text-slate-500">
                              {user.username === username ? '当前用户' : '普通用户'}
                            </p>
                          </div>
                        </div>
                        {user.username !== username && (
                          <button 
                            onClick={() => deleteUser(user)}
                            className="p-2 text-slate-500 hover:text-red-500 hover:bg-red-500/10 rounded transition-colors"
                            title="删除用户"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* System Info */}
              <div className="bg-[#181c22] p-6 rounded-xl border border-white/5 shadow-lg lg:col-span-2">
                <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  系统概览
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 bg-[#10141a] border border-white/5 rounded-lg text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">历史记录</p>
                    <p className="text-2xl font-bold text-[#00daf3]">{history.length}</p>
                  </div>
                  <div className="p-4 bg-[#10141a] border border-white/5 rounded-lg text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">总检测量</p>
                    <p className="text-2xl font-bold text-white">{history.reduce((acc, r) => acc + r.stats.total, 0)}</p>
                  </div>
                  <div className="p-4 bg-[#10141a] border border-white/5 rounded-lg text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">模型状态</p>
                    <div className="flex items-center justify-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${modelLoaded ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                      <p className={`text-sm font-bold ${modelLoaded ? 'text-emerald-400' : 'text-slate-500'}`}>{modelLoaded ? '就绪' : '未加载'}</p>
                    </div>
                  </div>
                  <div className="p-4 bg-[#10141a] border border-white/5 rounded-lg text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">API 状态</p>
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      <p className="text-sm font-bold text-emerald-400">在线</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {/* History Modal */}
        {selectedRecord && (
          <div 
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 md:p-8" 
            onClick={() => setSelectedRecord(null)}
          >
            <div 
              className="bg-[#10141a] border border-white/10 rounded-2xl overflow-hidden max-w-5xl w-full max-h-full flex flex-col shadow-2xl scale-in" 
              onClick={e => e.stopPropagation()}
            >
              <div className="p-4 border-b border-white/5 flex justify-between items-center bg-[#181c22]">
                <h3 className="text-white font-bold flex items-center gap-2">
                  <History className="w-5 h-5 text-[#00daf3]" />
                  识别记录 - {selectedRecord.time}
                </h3>
                <button 
                  onClick={() => setSelectedRecord(null)} 
                  className="text-slate-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/5"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 overflow-hidden bg-black p-4 flex flex-col items-center justify-center min-h-[200px] gap-2">
                {selectedRecord.image
                  ? <img
                      src={selectedRecord.image}
                      alt="Snapshot"
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                  : <div className="flex flex-col items-center justify-center text-white/30 gap-3">
                      <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 10l4.553-2.069A1 1 0 0121 8.82V15.18a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" /></svg>
                      <span className="text-sm">视频记录（无截图）</span>
                    </div>
                }
              </div>
              
              {/* Detections Table */}
              <div className="h-48 bg-[#10141a] overflow-y-auto custom-scrollbar border-t border-white/5 p-4">
                <h4 className="text-white text-sm font-bold mb-3 flex items-center gap-2">
                  <Box className="w-4 h-4 text-[#00daf3]" />
                  检测目标详情
                </h4>
                <table className="w-full text-left text-xs text-slate-400">
                  <thead className="text-slate-500 uppercase bg-white/5 sticky top-0">
                    <tr>
                      <th className="p-2 rounded-l-lg font-medium">类别</th>
                      <th className="p-2 font-medium">置信度</th>
                      <th className="p-2 rounded-r-lg font-medium">坐标 [x1, y1, x2, y2]</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedRecord.detections.map((det, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                        <td className="p-2 text-[#00daf3] font-medium">{det.className}</td>
                        <td className="p-2">{(det.score * 100).toFixed(1)}%</td>
                        <td className="p-2 font-mono">{det.box.map(v => Math.round(v)).join(', ')}</td>
                      </tr>
                    ))}
                    {selectedRecord.detections.length === 0 && (
                      <tr><td colSpan={3} className="p-4 text-center text-slate-500">未检测到目标</td></tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="p-4 bg-[#181c22] border-t border-white/5 flex items-center justify-between">
                <div className="grid grid-cols-3 gap-4 text-center flex-1">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">目标数</p>
                    <p className="text-2xl font-bold text-[#00daf3]">{selectedRecord.stats.total}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">FPS</p>
                    <p className="text-2xl font-bold text-white">{selectedRecord.stats.fps}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">平均置信度</p>
                    <p className="text-2xl font-bold text-white">{(selectedRecord.stats.avgConf * 100).toFixed(1)}%</p>
                  </div>
                </div>
                <button
                  onClick={(e) => { deleteRecord(selectedRecord, e); }}
                  className="ml-4 flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg hover:bg-red-500/20 transition-colors text-sm font-medium"
                >
                  <Trash2 className="w-4 h-4" />
                  删除
                </button>
              </div>
            </div>
          </div>
        )}

        <ToastContainer />

      </main>
    </div>
  );
}
