-- ============================================
-- YOLO Vision Dashboard - Supabase Schema
-- 在 Supabase SQL Editor 中执行此文件
-- ============================================

-- 用户配置表（扩展 Supabase auth.users）
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  role TEXT DEFAULT 'operator',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 检测历史记录表
CREATE TABLE IF NOT EXISTS public.detection_records (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  time TEXT NOT NULL,
  image TEXT,
  total_detections INTEGER DEFAULT 0,
  fps REAL DEFAULT 0,
  avg_confidence REAL DEFAULT 0,
  detections JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- Row Level Security (RLS)
-- ============================================

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detection_records ENABLE ROW LEVEL SECURITY;

-- profiles: 认证用户可查看所有 profile
CREATE POLICY "Profiles are viewable by authenticated users"
  ON public.profiles FOR SELECT
  TO authenticated
  USING (true);

-- profiles: 用户只能更新自己的 profile
CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  TO authenticated
  USING (auth.uid() = id);

-- profiles: 允许 insert（注册时用）
CREATE POLICY "Users can insert own profile"
  ON public.profiles FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = id);

-- detection_records: 用户只能查看自己的记录
CREATE POLICY "Users can view own records"
  ON public.detection_records FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

-- detection_records: 用户只能插入自己的记录
CREATE POLICY "Users can insert own records"
  ON public.detection_records FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- detection_records: 用户只能删除自己的记录
CREATE POLICY "Users can delete own records"
  ON public.detection_records FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- ============================================
-- 索引
-- ============================================

CREATE INDEX IF NOT EXISTS idx_detection_records_user_id
  ON public.detection_records(user_id);

CREATE INDEX IF NOT EXISTS idx_detection_records_created_at
  ON public.detection_records(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_profiles_username
  ON public.profiles(username);

-- ============================================
-- 触发器: 注册时自动创建 profile (可选，也可由后端 API 创建)
-- ============================================

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, username)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1))
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 删除已存在的触发器（如有）
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- 创建触发器
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
