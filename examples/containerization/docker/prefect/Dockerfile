FROM python:3.9-slim

RUN pip install prefect

EXPOSE 4200

COPY ./start_agent.sh /start_agent.sh
RUN chmod +x /start_agent.sh

CMD ["/start_agent.sh"]
