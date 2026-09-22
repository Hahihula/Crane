import { useMemo, useState } from 'react'

const messages = {
  zh: {
    newChat: '新对话', history: '历史会话', untitled: '新对话', deleteChat: '删除会话', closeHistory: '关闭历史记录', localHistory: '历史记录仅保存在当前浏览器中。', emptyTitle: '今天想聊些什么？', emptyHint: 'Crane 会流式生成回答，你可以随时停止。', placeholder: '给 Crane 发送消息', send: '发送', stop: '停止生成', attach: '添加图片', removeAttachment: '移除附件', thinking: '正在思考…', thought: '思考过程', copied: '已复制', copy: '复制', retry: '重新生成', stopped: '已停止生成', failed: '生成失败', latest: '回到最新消息', user: '你', assistant: 'AI', disclaimer: 'Crane 可能会犯错，请核查重要信息。', empty: '模型未返回文本', requestFailed: '请求失败', browserStream: '浏览器不支持流式响应', language: 'English', lengthStop: '已达到生成长度限制', clear: '清空对话', image: '图片', generating: '正在生成', stopReason: '回答完成', interrupted: '已由你停止', enterHint: 'Enter 发送 · Shift + Enter 换行', settings: '设置', temperature: '温度', topP: 'Top P', maxTokens: '最大生成 Token', unlimited: '不限制', model: '模型', collapseSidebar: '收起侧栏', expandSidebar: '展开侧栏', closeSettings: '关闭设置',
  },
  en: {
    newChat: 'New chat', history: 'History', untitled: 'New chat', deleteChat: 'Delete conversation', closeHistory: 'Close history', localHistory: 'History is stored only in this browser.', emptyTitle: 'What can I help with?', emptyHint: 'Crane streams responses and can be stopped at any time.', placeholder: 'Message Crane', send: 'Send', stop: 'Stop generating', attach: 'Attach image', removeAttachment: 'Remove attachment', thinking: 'Thinking…', thought: 'Reasoning', copied: 'Copied', copy: 'Copy', retry: 'Regenerate', stopped: 'Generation stopped', failed: 'Generation failed', latest: 'Jump to latest', user: 'You', assistant: 'AI', disclaimer: 'Crane can make mistakes. Check important information.', empty: 'The model returned no text', requestFailed: 'Request failed', browserStream: 'Streaming is not supported by this browser', language: '中文', lengthStop: 'The generation length limit was reached', clear: 'Clear conversation', image: 'Image', generating: 'Generating', stopReason: 'Response complete', interrupted: 'Stopped by you', enterHint: 'Enter to send · Shift + Enter for a new line', settings: 'Settings', temperature: 'Temperature', topP: 'Top P', maxTokens: 'Max output tokens', unlimited: 'Unlimited', model: 'Model', collapseSidebar: 'Collapse sidebar', expandSidebar: 'Expand sidebar', closeSettings: 'Close settings',
  },
} as const

export type Locale = keyof typeof messages
export type Translations = typeof messages.zh | typeof messages.en

export function useI18n() {
  const [locale, setLocale] = useState<Locale>(() => navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en')
  return useMemo(() => ({ locale, t: messages[locale] as Translations, toggle: () => setLocale(value => value === 'zh' ? 'en' : 'zh') }), [locale])
}
