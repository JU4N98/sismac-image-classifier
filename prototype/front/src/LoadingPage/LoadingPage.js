export const LoadingPage = () => {
    return (
      <div style={{ textAlign: 'center', marginTop: '20%' }}>
        <h2>Loading...</h2>
        <div className="spinner-border" role="status">
          <span className="sr-only">Loading...</span>
        </div>
      </div>
    );
  };