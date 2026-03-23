#!/bin/bash
set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function show_help {
    echo -e "${BLUE}OpenRingDetector Management Script${NC}"
    echo ""
    echo "Usage: ./ring-detector.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start         Start all services"
    echo "  stop          Stop all services"
    echo "  restart       Restart all services (picks up .env changes)"
    echo "  build         Build/rebuild Docker images from source"
    echo "  deploy        Build images and restart all services"
    echo "  status        Show container status"
    echo "  logs [svc]    View logs (all or specific service: db, api, frontend, ntfy, ollama)"
    echo "  health        Check service health"
    echo "  shell [svc]   Open shell in container (default: ring-api)"
    echo "  clean         Remove all containers and volumes (CAUTION: destroys DB data)"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./ring-detector.sh start"
    echo "  ./ring-detector.sh deploy          # Rebuild and restart after code changes"
    echo "  ./ring-detector.sh restart         # Restart after .env changes"
    echo "  ./ring-detector.sh logs ring-api"
    echo "  ./ring-detector.sh shell ring-api"
    echo ""
}

check_environment() {
    if [ ! -f .env ]; then
        echo -e "${RED}❌ .env file not found${NC}"
        echo "Copy .env.example to .env and fill in your values."
        exit 1
    fi

    if [ ! -f docker-compose.yml ]; then
        echo -e "${RED}❌ docker-compose.yml not found${NC}"
        exit 1
    fi
}

show_access_info() {
    source .env 2>/dev/null || true
    echo -e "${GREEN}Services:${NC}"
    echo "  Dashboard:  http://localhost:${FRONTEND_PORT:-9554}"
    echo "  API:        http://localhost:${API_PORT:-9553}/docs"
    echo "  ntfy:       http://localhost:${NTFY_PORT:-9552}"
    echo "  Ollama:     http://localhost:${OLLAMA_PORT:-9551}"
    echo "  DB:         localhost:${DB_PORT:-9550}"
    echo ""
    echo -e "${YELLOW}Start ring-watch from the Dashboard to begin monitoring.${NC}"
    echo ""
}

case "${1:-help}" in
    start)
        check_environment
        echo -e "${YELLOW}Starting OpenRingDetector...${NC}"
        docker compose up -d
        echo -e "${GREEN}✅ OpenRingDetector started${NC}"
        echo ""
        show_access_info
        ;;
    stop)
        check_environment
        echo -e "${YELLOW}Stopping OpenRingDetector...${NC}"
        docker compose down
        echo -e "${GREEN}✅ OpenRingDetector stopped${NC}"
        ;;
    restart)
        check_environment
        echo -e "${YELLOW}Restarting OpenRingDetector (picks up .env changes)...${NC}"
        docker compose down
        docker compose up -d
        echo -e "${GREEN}✅ OpenRingDetector restarted${NC}"
        echo ""
        show_access_info
        ;;
    build)
        check_environment
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker compose build
        echo -e "${GREEN}✅ Build complete${NC}"
        ;;
    deploy)
        check_environment
        echo -e "${YELLOW}Building and deploying OpenRingDetector...${NC}"
        docker compose build
        docker compose down
        docker compose up -d
        echo -e "${GREEN}✅ OpenRingDetector deployed${NC}"
        echo ""
        show_access_info
        ;;
    status)
        check_environment
        echo -e "${BLUE}Container Status:${NC}"
        docker compose ps
        ;;
    logs)
        check_environment
        service=${2:-}
        if [ -z "$service" ]; then
            echo -e "${BLUE}Showing all logs (Ctrl+C to exit):${NC}"
            docker compose logs -f
        else
            echo -e "${BLUE}Showing logs for $service (Ctrl+C to exit):${NC}"
            docker compose logs -f "$service"
        fi
        ;;
    health)
        check_environment
        source .env 2>/dev/null || true
        echo -e "${BLUE}Health Check:${NC}"
        echo ""
        docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "Service Health:"

        if curl -s "http://localhost:${API_PORT:-9553}/health" > /dev/null 2>&1; then
            echo "  ✅ ring-api:    Healthy"
        else
            echo "  ❌ ring-api:    Unhealthy (http://localhost:${API_PORT:-9553})"
        fi

        if curl -s "http://localhost:${FRONTEND_PORT:-9554}" > /dev/null 2>&1; then
            echo "  ✅ ring-frontend: Healthy"
        else
            echo "  ❌ ring-frontend: Unhealthy (http://localhost:${FRONTEND_PORT:-9554})"
        fi

        if docker compose exec -T db pg_isready -U "${DB_USER:-postgres}" > /dev/null 2>&1; then
            echo "  ✅ db:          Healthy"
        else
            echo "  ❌ db:          Unhealthy"
        fi

        if curl -s "http://localhost:${NTFY_PORT:-9552}/v1/health" > /dev/null 2>&1; then
            echo "  ✅ ntfy:        Healthy"
        else
            echo "  ❌ ntfy:        Unhealthy (http://localhost:${NTFY_PORT:-9552})"
        fi

        if curl -s "http://localhost:${OLLAMA_PORT:-9551}/api/version" > /dev/null 2>&1; then
            echo "  ✅ ollama:      Healthy"
        else
            echo "  ❌ ollama:      Unhealthy (http://localhost:${OLLAMA_PORT:-9551})"
        fi
        echo ""
        ;;
    shell)
        check_environment
        service=${2:-ring-api}
        echo -e "${BLUE}Opening shell in $service...${NC}"
        docker compose exec "$service" /bin/bash || docker compose exec "$service" /bin/sh
        ;;
    clean)
        check_environment
        echo -e "${RED}⚠️  WARNING: This will remove ALL containers and volumes including the database!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Removing all data...${NC}"
            docker compose down -v
            echo -e "${GREEN}✅ All containers and volumes removed${NC}"
        else
            echo -e "${GREEN}✅ Cancelled${NC}"
        fi
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
