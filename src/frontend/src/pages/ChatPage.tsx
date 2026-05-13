import { useState } from "react";
import ChatInput, { type StagedAsset } from "../components/ChatInput";
import MessageList from "../components/MessageList";
import ModelPicker from "../components/ModelPicker";
import { useAuth } from "../hooks/useAuth";
import { useChat } from "../hooks/useChat";

export default function ChatPage() {
  const { logout } = useAuth();
  const chat = useChat();
  const [model, setModel] = useState<string | null>(null);
  const [topK, setTopK] = useState(8);
  const [numChoices, setNumChoices] = useState(3);

  function onSend(message: string, attachments: StagedAsset[]) {
    void chat.send({
      message,
      attachments: attachments.map((a) => ({
        asset_id: a.asset_id,
        display_name: a.display_name,
        preview_url: a.preview_url ?? null,
      })),
      model,
      topK,
      numChoices,
    });
  }

  return (
    <div className="chat-shell">
      <div className="chat-header">
        <div className="logo">IP</div>
        <div>
          <div className="title">AI Imaging Agent</div>
          <div className="sub">Imaging Plaza · EPFL</div>
        </div>
        <div className="spacer" />
        <button className="btn-logout" onClick={() => void logout()}>
          Sign out
        </button>
      </div>

      <div style={{ padding: "10px var(--space) 0" }}>
        <ModelPicker
          value={model}
          onChange={setModel}
          topK={topK}
          onTopK={setTopK}
          numChoices={numChoices}
          onNumChoices={setNumChoices}
        />
      </div>

      <MessageList
        turns={chat.turns}
        busy={chat.busy}
        onApprove={() => void chat.approve()}
        onDecline={() => void chat.decline()}
        onConfirmDemo={() => void chat.confirmDemo()}
      />

      <ChatInput
        sessionId={chat.sessionId}
        onSessionId={chat.setSessionId}
        onSend={onSend}
        busy={chat.busy}
      />
    </div>
  );
}
