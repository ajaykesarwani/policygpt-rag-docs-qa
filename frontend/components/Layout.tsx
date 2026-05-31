// import React from "react";

// type Props = {
//   children: React.ReactNode;
// };

// export default function Layout({ children }: Props) {
//   return (
//     <>
//       <header className="app-header">
//         <div className="app-header__title-block">
//           <div className="app-header__badge">
//             <span>PolicyGPT</span>
//             <span>· An AI assistant that answers company policy questions using only your internal documents.</span>
//           </div>
//           <h1 className="app-header__title">RAG Document Q&A workspace</h1>
//           <p className="app-header__subtitle">
//             Upload internal PDFs or text files, then ask grounded questions. The
//             assistant answers strictly from retrieved context, and shows you the
//             source snippets it used.
//           </p>
//         </div>
//         <div className="app-header__pill">
//           Built with Groq · local embeddings · FastAPI · Next.js
//         </div>
//       </header>

//       <main className="app-main-grid">{children}</main>
//     </>
//   );
// }

import React from "react";

type Props = {
  children: React.ReactNode;
};

export default function Layout({ children }: Props) {
  return (
    <div className="app-shell">
      <div className="app-shell__inner">
        <header className="app-header">
          <div className="app-header__title-block">
            <div className="app-header__badge">
              <span className="app-header__badge-name">PolicyGPT</span>
              <span className="app-header__badge-tagline">
                An AI assistant that answers company policy questions using only your internal documents.
              </span>
            </div>

            <h1 className="app-header__title">RAG Document Q&amp;A workspace</h1>

            <p className="app-header__subtitle">
              Upload internal PDFs or text files, then ask grounded questions.
              The assistant answers strictly from retrieved context, and shows
              you the source snippets it used.
            </p>
          </div>

          <div className="app-header__pill">
            RAG pipeline · Built with Groq · local embeddings · FastAPI · Next.js
          </div>
        </header>

        <main className="app-main-grid">{children}</main>
      </div>
    </div>
  );
}