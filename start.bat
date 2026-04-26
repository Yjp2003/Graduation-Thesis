@echo off
chcp 65001 >nul
title YOLO Vision Dashboard

echo ============================================
echo  YOLO Vision Dashboard 启动脚本
echo ============================================
echo.

:: 启动后端
echo [1/2] 启动后端 FastAPI (端口 3001)...
start "YOLO Backend" cmd /k "cd /d d:\YOLO\server && python main.py"

:: 等待后端启动
timeout /t 2 /nobreak >nul

:: 启动前端
echo [2/2] 启动前端 Vite React (端口 3000)...
start "YOLO Frontend" cmd /k "cd /d d:\YOLO && npm run dev"

echo.
echo ============================================
echo  服务启动中，请稍候...
echo  前端地址: http://localhost:3000
echo  后端地址: http://localhost:3001
echo  API文档:  http://localhost:3001/docs
echo ============================================
echo.
pause
