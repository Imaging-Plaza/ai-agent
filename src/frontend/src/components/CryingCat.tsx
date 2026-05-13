/** Fallback shown when a resumed conversation references attachments that
 *  the server can no longer serve (TTL eviction, container restart, …).
 *
 *  Pure CSS art so it lands instantly without an extra HTTP roundtrip.
 */

export default function CryingCat() {
  return (
    <div className="cat-card">
      <pre className="cat-art mono" aria-hidden>
        {String.raw`
   /\_/\
  ( ;_; )
  / >🖼️<\
        `}
      </pre>
      <div className="cat-msg">
        <strong>oh no, your image ran away —</strong>
        <p>
          uploads are kept on the server for a few hours, and yours has timed
          out since you last chatted. drop the image back in to keep going.
        </p>
      </div>
    </div>
  );
}
