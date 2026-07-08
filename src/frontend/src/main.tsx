import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./hooks/useAuth";
import { ConversationsProvider } from "./hooks/useConversations";
import { ThemeProvider } from "./hooks/useTheme";
import { TranscriptionProvider } from "./hooks/useTranscription";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <ConversationsProvider>
            <TranscriptionProvider>
              <App />
            </TranscriptionProvider>
          </ConversationsProvider>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  </React.StrictMode>
);
