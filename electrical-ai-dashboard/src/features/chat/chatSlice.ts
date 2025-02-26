// src/features/chat/chatSlice.ts
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatState {
  conversation: ChatMessage[];
}

const initialState: ChatState = {
  conversation: [],
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    addMessage(state, action: PayloadAction<ChatMessage>) {
      state.conversation.push(action.payload);
    },
    updateLastMessage(state, action: PayloadAction<string>) {
      if (state.conversation.length > 0) {
        const lastMsg = state.conversation[state.conversation.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.content = action.payload;
        }
      }
    },
    removeLastAssistantIfEmpty(state) {
      if (
        state.conversation.length > 0 &&
        state.conversation[state.conversation.length - 1].role === 'assistant' &&
        state.conversation[state.conversation.length - 1].content.trim() === ""
      ) {
        state.conversation.pop();
      }
    },
    setConversation(state, action: PayloadAction<ChatMessage[]>) {
      state.conversation = action.payload;
    },
    clearConversation(state) {
      state.conversation = [];
    },
  },
});

export const { addMessage, updateLastMessage, removeLastAssistantIfEmpty, setConversation, clearConversation } = chatSlice.actions;
export default chatSlice.reducer;