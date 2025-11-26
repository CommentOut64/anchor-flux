/**
 * Vue Router 配置
 */
import { createRouter, createWebHistory } from 'vue-router'

// 路由配置
const routes = [
  {
    path: '/',
    redirect: '/store-test'
  },
  {
    // 阶段1测试页面 - 用于验证服务层功能
    path: '/test',
    name: 'TestView',
    component: () => import('@/views/TestView.vue'),
    meta: { title: '服务测试' }
  },
  {
    // 阶段2测试页面 - 用于验证状态管理层功能
    path: '/store-test',
    name: 'StoreTestView',
    component: () => import('@/views/StoreTestView.vue'),
    meta: { title: 'Store测试' }
  },
  {
    // 任务列表页 (阶段5实现)
    path: '/tasks',
    name: 'TaskList',
    component: () => import('@/views/TaskListView.vue'),
    meta: { title: '任务列表' }
  },
  {
    // 编辑器页面 (阶段5实现)
    path: '/editor/:jobId',
    name: 'Editor',
    component: () => import('@/views/EditorView.vue'),
    props: true,
    meta: { title: '字幕编辑' }
  }
]

// 创建路由实例
const router = createRouter({
  history: createWebHistory(),
  routes
})

// 全局前置守卫 - 设置页面标题
router.beforeEach((to, from, next) => {
  document.title = to.meta.title ? `${to.meta.title} - VideoSRT` : 'VideoSRT'
  next()
})

export default router
