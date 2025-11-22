import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import "./style.css";
import axios from "axios";
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import * as ElementPlusIconsVue from '@element-plus/icons-vue';
import { useModelStore } from './stores/modelStore.js';

// 移除baseURL设置，让前端直接使用相对路径
// const base = window.__API_BASE__ || "/api";
// axios.defaults.baseURL = base;

const app = createApp(App);

// 创建Pinia实例
const pinia = createPinia();

// 使用Pinia（必须在挂载App之前）
app.use(pinia);

// 使用Element Plus
app.use(ElementPlus);

// 注册所有图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component);
}

// 挂载应用
app.mount("#app");

// 应用挂载后，初始化模型管理器（建立全局SSE连接）
const modelStore = useModelStore();
modelStore.initialize().catch(error => {
  console.error('[App] 模型管理器初始化失败:', error);
});
