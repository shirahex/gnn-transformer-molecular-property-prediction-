# Molecular Property Prediction - Frontend

React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui frontend for the Molecular AI platform.

## Quick Start

```bash
cd frontend
npm install
npm run build   # Production build
npm start       # Development server
```

## Development

The development server runs at `http://localhost:3000` and proxies API requests to the backend at `http://localhost:8000`.

### Available Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Start dev server with hot reload |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview production build locally |

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Layout.tsx       # Sidebar navigation layout
│   ├── DatasetCard.tsx  # Dataset statistics card
│   ├── PredictionForm.tsx  # SMILES input form
│   ├── MoleculeViewer.tsx  # 2D molecule renderer
│   ├── BenchmarkChart.tsx  # Results visualization
│   ├── TrainingMonitor.tsx # Live training tracker
│   └── PosterTemplate.tsx  # Poster generator
├── pages/               # Route-level pages
│   ├── Dashboard.tsx
│   ├── Predictor.tsx
│   ├── Training.tsx
│   ├── Results.tsx
│   ├── Explain.tsx
│   └── Poster.tsx
├── services/
│   └── api.ts           # Axios API client
├── types/
│   └── index.ts         # TypeScript interfaces
├── App.tsx              # Router setup
└── main.tsx             # Entry point
```

## Styling

- **Tailwind CSS** for utility-first styling
- **shadcn/ui** components with dark theme
- Custom scrollbar and print styles in `index.css`

## API Configuration

Update `src/services/api.ts` to change the backend URL:

```typescript
const API_BASE = "http://localhost:8000"; // Change if backend runs elsewhere
```
