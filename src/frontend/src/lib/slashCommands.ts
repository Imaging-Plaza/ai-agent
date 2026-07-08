/** Slash-command parser used by the chat composer.
 *
 *  Recognised commands (case-insensitive, alias-friendly):
 *    /img | /image  <id | name-substring | url>   →  image embed
 *    /audio | /sound <url>                        →  inline <audio> player
 *    /video | /clip  <url>                        →  inline <video> player
 *    /youtube | /yt  <video-id | watch-url>       →  YouTube iframe embed
 *    /embed | /iframe <url>                       →  generic iframe
 *    /help                                        →  list commands
 *
 *  Anything else falls through and is sent to the agent as a normal message.
 */

export type ParsedCommand =
  | { kind: "image"; query: string }
  | { kind: "audio"; url: string }
  | { kind: "video"; url: string }
  | { kind: "youtube"; videoId: string }
  | { kind: "iframe"; url: string }
  | { kind: "help" }
  | { kind: "unknown"; raw: string };

const ALIASES: Record<string, string> = {
  img: "image",
  image: "image",
  audio: "audio",
  sound: "audio",
  video: "video",
  clip: "video",
  youtube: "youtube",
  yt: "youtube",
  embed: "iframe",
  iframe: "iframe",
  help: "help",
  "?": "help",
};

export function isSlashCommand(text: string): boolean {
  return /^\s*\//.test(text);
}

export function parseSlashCommand(text: string): ParsedCommand | null {
  const m = /^\s*\/(\S+)\s*(.*)$/s.exec(text);
  if (!m) return null;
  const verb = m[1].toLowerCase();
  const rest = m[2].trim();
  const canonical = ALIASES[verb];
  if (!canonical) {
    return { kind: "unknown", raw: text.trim() };
  }
  switch (canonical) {
    case "image":
      return { kind: "image", query: rest };
    case "audio":
      return { kind: "audio", url: rest };
    case "video":
      return { kind: "video", url: rest };
    case "youtube":
      return { kind: "youtube", videoId: extractYouTubeId(rest) };
    case "iframe":
      return { kind: "iframe", url: rest };
    case "help":
      return { kind: "help" };
  }
  return { kind: "unknown", raw: text.trim() };
}

function extractYouTubeId(s: string): string {
  if (!s) return "";
  // Already an 11-char id?
  if (/^[A-Za-z0-9_-]{11}$/.test(s)) return s;
  try {
    const u = new URL(s);
    if (u.hostname.includes("youtu.be")) {
      return u.pathname.replace(/^\//, "");
    }
    if (u.hostname.includes("youtube.com")) {
      const v = u.searchParams.get("v");
      if (v) return v;
      // Shorts / embed URLs: /shorts/ID or /embed/ID
      const parts = u.pathname.split("/").filter(Boolean);
      const last = parts.pop();
      return last || "";
    }
  } catch {
    /* not a URL */
  }
  return s;
}

export const HELP_TEXT = [
  "/img <name | id-prefix | url>   embed an image from this session or a URL",
  "/audio <url>                    inline audio player",
  "/video <url>                    inline video player",
  "/youtube <id | watch-url>       YouTube embed",
  "/embed <url>                    generic iframe (PDFs, web pages)",
  "/help                           this list",
].join("\n");
