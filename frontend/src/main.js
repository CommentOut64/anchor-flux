import { createApp } from "vue";
import App from "./App.vue";
import "./style.css";
import axios from "axios";
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import * as ElementPlusIconsVue from '@element-plus/icons-vue';

// 移除baseURL设置，让前端直接使用相对路径
// const base = window.__API_BASE__ || "/api";
// axios.defaults.baseURL = base;

const app = createApp(App);

// 使用Element Plus
app.use(ElementPlus);

// 注册所有图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component);
}

app.mount("#app");
