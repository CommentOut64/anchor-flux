import { createApp } from "vue";
import App from "./App.vue";
import "./style.css";
import axios from "axios";

// 移除baseURL设置，让前端直接使用相对路径
// const base = window.__API_BASE__ || "/api";
// axios.defaults.baseURL = base;

createApp(App).mount("#app");
