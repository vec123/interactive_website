import LatentExplorer from "./GPLVM_latent_explorer";
import MethodDescription from "./GPLVM";
import "katex/dist/katex.min.css";
function App() {
  return (
    <div>
      <MethodDescription/>
      <LatentExplorer />
    </div>
  );
}

export default App;
